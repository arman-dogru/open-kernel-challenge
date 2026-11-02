"""Build the bf16 v32 runs files from runs/v6_router_stacked.py (board 1.115x) + tools/bf16_v32_stream.cu +
tools/bf16_v32_cfg.py. Run from the repo root.

  python3 tools/build_bf16_v32.py --diag      -> muse_bf16_proj/runs/v32d_bench.py (DIAGNOSTIC: v6's kernel_fn
                                                 unchanged + untimed in-prepare timing of every v32 candidate)
  python3 tools/build_bf16_v32.py ROUTE.json [name] -> muse_bf16_proj/runs/<name or v32_routed>.py: v6 + v32 on the
                                                 routed (layer, m bucket) pairs; ROUTE.json {"key:bucket": [[KS, G], [NW, ST, SPS, RT]]}
"""
import json, re, sys

DIAG = "--diag" in sys.argv
route, name = {}, "v32_routed"
if not DIAG:
    route = json.load(open(sys.argv[1]))
    if len(sys.argv) > 2:
        name = sys.argv[2]
SRC = "muse_bf16_proj/runs/v6_router_stacked.py"
s = open(SRC).read()
cu = open("tools/bf16_v32_stream.cu").read()
cfgsrc = open("tools/bf16_v32_cfg.py").read()
assert '"""' not in cu


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


_REN = [("ALL_INSTANCES", "V32_INSTANCES"), ("SMEM_MAX", "V32_SMEM_MAX"), ("SMEM_MIN", "V32_SMEM_MIN"),
        ("SMS", "V32_SMS"), ("LAYERS", "V32_LAYERS"), ("PLANS", "V32_PLANS"), ("TUNES", "V32_TUNES"), ("NTS", "V32_NTS"),
        ("smem", "_v32_smem"), ("valid", "_v32_valid"), ("instance", "_v32_instance"), ("name_expr", "_v32_name"),
        ("candidates", "_v32_candidates"), ("launch", "_v32_launch")]


def ren(py):
    """v32's Python names get a v32 prefix in the runs file (v6 already has CONFIGS etc.)."""
    for a, b in _REN:
        py = re.sub(r"\b%s\b" % a, b, py)
    return py


cfg_body = cfgsrc.split('"""', 2)[2]  # drop the module docstring
if DIAG:
    inst_expr = "V32_INSTANCES"
else:
    inst_expr = "sorted({_v32_instance(k, nt, p, t) for (k, b), (p, t) in V32_ROUTE.items() for nt in _v32_nts(b)})"

HOST = r'''

# ---------------- v32: TMA-bulk weight-streaming CUDA kernel (tools/bf16_v32_stream.cu, tools/bf16_v32_cfg.py) -------
_V32_SRC = r"""
''' + cu + r'''"""
# tools/bf16_v32_cfg.py (body; names prefixed _v32 / V32_)
''' + ren(cfg_body) + ren(r'''

_V32 = {}  # "cubin", "lowered": {inst: name}; device -> (module, {inst: function})


def _v32_nts(bucket):
    return (1,) if bucket <= 8 else ((2,) if bucket <= 16 else (4,))


def _v32_insts():
    return ''' + inst_expr + r'''


def _v32_compile():
    """NVRTC (cuda.bindings, in-process) -> one sm_90a cubin with the needed instantiations. Host only."""
    if "cubin" not in _V32:
        from cuda.bindings import nvrtc
        insts = _v32_insts()
        err, prog = nvrtc.nvrtcCreateProgram(_V32_SRC.encode(), b"bf16_v32.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        for inst in insts:
            nvrtc.nvrtcAddNameExpression(prog, name_expr(inst).encode())
        opts = [b"-arch=sm_90a", b"-std=c++17"]
        (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="ignore"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        lowered = {}
        for inst in insts:
            _, low = nvrtc.nvrtcGetLoweredName(prog, name_expr(inst).encode())
            lowered[inst] = low.decode()
        _V32["cubin"], _V32["lowered"] = cubin, lowered
    return _V32["cubin"]


def _v32_load(dev):
    """CuPy functions on device dev (compile + load in prepare(), never inside graph capture)."""
    if dev not in _V32:
        import cupy as cp
        cubin = _v32_compile()
        with cp.cuda.Device(dev):
            mod = cp.cuda.function.Module()
            mod.load(cubin)
            fns = {}
            drv = cp.cuda.driver
            for inst, low in _V32["lowered"].items():
                f = mod.get_function(low)
                drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8),
                                     SMEM_MAX)
                fns[inst] = f
        _V32[dev] = (mod, fns)
    return _V32[dev][1]


def _v32_repack(w):
    """[n, K] BF16 -> A-fragment order [n/16][K/32][s][lane][8 bf16] (lossless permutation, same bytes)."""
    n, K = w.shape
    t = w.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2)  # rt, h, g, p, t, s, e, q
    return t.permute(0, 3, 5, 2, 4, 6, 1, 7).contiguous()  # rt, p, s, g, t, e, h, q


def _v32_router_rows(hi, lo):
    """[256, K] hi, lo -> [512, K]: rowtile 2c = hi rows 16c.., rowtile 2c+1 = lo rows 16c.. (summed by HL = 2)."""
    n, K = hi.shape
    return torch.stack([hi.view(n // 16, 16, K), lo.view(n // 16, 16, K)], 1).reshape(2 * n, K)


def _v32(x, key, plan, tune, state):
    """out[m, n_out] = x . W^T on the v32 kernel (BF16 out; FP32 for the router)."""
    import cupy as cp
    import numpy as np
    m, K = x.shape
    rows, Kw, hl, f32, n_out = LAYERS[key]
    nt = 1 if m <= 8 else (2 if m <= 16 else 4)
    assert K == Kw and m <= 8 * nt and x.is_contiguous() and x.data_ptr() % 16 == 0
    inst = instance(key, nt, plan, tune)
    grid, threads, shm, wsn = launch(key, plan, tune, nt)
    wr = state["v32_" + ("unpooler" if key == "unpooler_s" else key)]
    ws, cnt = state["v32_ws"], state["v32_cnt"]
    assert wsn <= ws.numel()
    out = torch.empty((m, n_out), device=x.device, dtype=torch.float32 if f32 else torch.bfloat16)
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    f = _v32_load(dev)[inst]
    c0 = _v32_cnt_off()[(key, tuple(plan))]
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
        f((grid,), (threads,), (np.uint64(wr.data_ptr()), np.uint64(x.data_ptr()), np.int32(K),
                                np.uint64(out.data_ptr()), np.int32(m), np.int32(n_out), np.int32(K),
                                np.int32(plan[1]), np.int32(plan[0]), np.uint64(ws.data_ptr()),
                                np.uint64(cnt.data_ptr() + 4 * c0)), shared_mem=shm)
    return out


def _v32_cnt_off():
    """Launch-counter slots: one run of grid-many u32 per (layer key, plan) (a grid's counters move in lockstep)."""
    if "cnt_off" not in _V32:
        off, o = {}, 0
        for key, plans in PLANS.items():
            rows = LAYERS[key][0]
            for ks, G in plans:
                off[(key, (ks, G))] = o
                o += rows // 16 // G * ks
        _V32["cnt_off"], _V32["cnt_n"] = off, o
    return _V32["cnt_off"]


def _v32_bucket(m):
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))

''')

PREP_ADD = '''    if torch.cuda.is_available() and weights["qkv_gate"].is_cuda:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            state["v32_" + name] = _v32_repack(weights[name].contiguous())
        state["v32_router"] = _v32_repack(_v32_router_rows(hi, lo))
        _v32_cnt_off()
        dev = weights["qkv_gate"].device
        state["v32_cnt"] = torch.zeros(_V32["cnt_n"], dtype=torch.int32, device=dev)  # launch counters, start at 0
        state["v32_ws"] = torch.empty(V32_SMS * 18 * 512, dtype=torch.float32, device=dev)  # split-K partials
        _v32_load(weights["qkv_gate"].device.index or 0)  # NVRTC compile + module load now, not in capture
'''

s = rep(s, "\ndef prepare(weights):", HOST + "\ndef prepare(weights):")
s = rep(s, '''    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    return state''', '''    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
''' + PREP_ADD + ('''    try:
        _diag_bench(state)  # v32d: diagnostic only, untimed
    except Exception as e:  # never fail the run
        print("v32d diag failed:", repr(e)[:600], flush=True)
''' if DIAG else "") + '''    return state''')

if DIAG:
    BENCH = r'''

_DIAG = {}


def _diag_bench(state):
    """One-time untimed sweep: per (layer, m) the v6 path (kernel_fn as checked), SGLang's cuBLAS path and every
    v32 candidate, each captured in a CUDA graph and replayed with a cold L2 (a 256 MB buffer read first), median of
    15. Also: an empty-ish graph (one tiny kernel) as the fixed-cost floor, and CTAs per SM (must be 1: the grid is <= 132 CTAs, one wave)."""
    import time
    if _DIAG or not torch.cuda.is_available():
        return
    _DIAG["done"] = True
    t0 = time.time()
    dev = state["qkv_gate"].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    tiny = torch.empty(16, device=dev)
    g = torch.Generator(device=dev).manual_seed(77)

    def timed(fn, reps=15):
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        ts = []
        for _ in range(reps):
            flush.view(torch.int32).sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    floor = timed(lambda: tiny.zero_())
    print(f"v32d diag: {torch.cuda.get_device_name()} SMs {torch.cuda.get_device_properties(dev).multi_processor_count}"
          f" floor(1 tiny kernel) {floor:.2f} us", flush=True)
    try:
        from cuda.bindings import driver as cu
        fns = _v32_load(dev.index or 0)
        seen = set()
        for key in LAYERS:
            for (plan, tune) in candidates(key, 1)[:1]:
                if plan[0] in seen:
                    continue
                seen.add(plan[0])
                inst = instance(key, 1, plan, tune)
                grid, threads, shm, _ = launch(key, plan, tune, 1)
                r = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(cu.CUfunction(fns[inst].ptr), threads, shm)
                print(f"  {key} KS {plan[0]} smem {shm >> 10} KB: CTAs/SM {r[1] if len(r) > 1 else r} (grid {grid})",
                      flush=True)
    except Exception as e:
        print("  occupancy query failed:", repr(e)[:200], flush=True)
    best = {}
    for key, (rows, K, hl, f32, n_out) in LAYERS.items():
        layer = "unpooler" if key == "unpooler_s" else key
        wbytes = rows * K * 2
        for m in ((1, 8) if key == "unpooler_s" else (1, 8, 16, 32)):
            if time.time() - t0 > 240:
                print("(time guard)", flush=True)
                break
            x = torch.randn((m, K), generator=g, device=dev).to(torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": n_out, "k": K, "layer": layer}}
            if key == "router":
                ref = x.float() @ state["router"].float().t()
                t_cb = timed(lambda: _router_stacked(x, state))
            else:
                ref = _baseline(x, layer, n_out, state).float()
                t_cb = timed(lambda: _baseline(x, layer, n_out, state))
            t_v6 = timed(lambda: kernel_fn(data, state))
            res = []
            for plan, tune in candidates(key, 1 if m <= 8 else (2 if m <= 16 else 4)):
                try:
                    out = _v32(x, key, plan, tune, state).float()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                    t = timed(lambda: _v32(x, key, plan, tune, state))
                except Exception as e:
                    print(f"{key} m {m} {plan} {tune} FAILED {repr(e)[:120]}", flush=True)
                    continue
                res.append((t, plan, tune, err))
            res.sort()
            line = [f"{key} m {m}: v6 {t_v6:.2f} base {t_cb:.2f} | v32 " + "; ".join(
                f"{p}{tu} {t:.2f} ({wbytes / max(t - floor, 0.1) / 1e6:.2f} TB/s) e{err:.0e}" for t, p, tu, err in res[:3])]
            if res:
                best[(key, m)] = (res[0], t_v6)
            print("\n".join(line), flush=True)
    print("v32d summary (key m: best v32 / v6 us):", flush=True)
    for (key, m), ((t, p, tu, err), t6) in best.items():
        print(f"  {key}:{m} {t:.2f} / {t6:.2f} {list(p)} {list(tu)} e{err:.0e}", flush=True)
    print(f"v32d diag: done in {time.time() - t0:.0f} s", flush=True)
'''
    s = s + ren(BENCH)
    hdr = '''# okc: problem=muse_bf16_proj
"""v32d (DIAGNOSTIC, not for the board): v6's kernel_fn unchanged, plus the v32 TMA-bulk weight-streaming CUDA
kernel (tools/bf16_v32_stream.cu) compiled in prepare() and an untimed one-time sweep of its candidates against v6
and cuBLAS for every BF16 layer x m <= 32 and the router (tools/build_bf16_v32.py --diag).

v6:'''
    dst = "muse_bf16_proj/runs/v32d_bench.py"
else:
    ROUTE_SRC = "V32_ROUTE = {\n" + "".join(
        f"    ({k.split(':')[0]!r}, {int(k.split(':')[1])}): ({tuple(v[0])}, {tuple(v[1])}),\n"
        for k, v in sorted(route.items())) + "}\n"
    s = rep(s, "\n# ---------------- v32:", "\n# (layer key, m bucket 1 / 8 / 16 / 32) -> v32 ((KS, G), (NW, ST, SPS, RT)), from the v32d diagnostic\n"
            + ROUTE_SRC + "\n# ---------------- v32:")
    s = rep(s, '''    m, k = x.shape
    if layer == "router" and m > TRITON_MAX_M:''', '''    m, k = x.shape
    if m <= 32 and x.is_cuda:
        key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
        r31 = V32_ROUTE.get((key, _v32_bucket(m)))
        if r31 is not None:
            return _v32(x.contiguous(), key, r31[0], r31[1], state)
    if layer == "router" and m > TRITON_MAX_M:''')
    hdr = '''# okc: problem=muse_bf16_proj
"""v32: v6 plus a TMA-bulk weight-streaming CUDA kernel (tools/bf16_v32_stream.cu: W repacked into mma.sync
A-fragment order and streamed by a producer warp with cp.async.bulk into a shared-memory ring, x's k-slice staged in
shared memory once per CTA, one CTA per SM on a balanced one-wave grid, split-K reduced in shared memory and then
through a global workspace with per-CTA launch counters (release/acquire, single writer: no atomics)) on the (layer, m) pairs in V32_ROUTE, picked from the v32d
diagnostic's per-shape timings vs v6 / cuBLAS. Everything else is v6. tools/build_bf16_v32.py.

v6:'''
    dst = f"muse_bf16_proj/runs/{name}.py"
s = rep(s, '# okc: problem=muse_bf16_proj\n"""', hdr + " ")
open(dst, "w").write(s)
print("wrote", dst)
