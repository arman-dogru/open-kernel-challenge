"""Build the bf16 v34 runs files from runs/v6_router_stacked.py (board 1.115x) + tools/bf16_v34_stream.cu +
tools/bf16_v34_cfg.py. Run from the repo root.

  python3 tools/build_bf16_v34.py --diag      -> muse_bf16_proj/runs/v34d_bench.py (DIAGNOSTIC: v6's kernel_fn
                                                 unchanged + untimed in-prepare timing of every v34 candidate)
  python3 tools/build_bf16_v34.py ROUTE.json [name] -> muse_bf16_proj/runs/<name or v34_routed>.py: v6 + v34 on the
                                                 routed (layer, m bucket) pairs; ROUTE.json {"key:bucket": [[KS, G], [NW, ST, SPS, RT]]}
"""
import json, re, sys

DIAG = "--diag" in sys.argv
DIAG_NAME = sys.argv[sys.argv.index("--diag") + 1] if DIAG and len(sys.argv) > sys.argv.index("--diag") + 1 else "v34d_bench"
route, name = {}, "v34_routed"
if not DIAG:
    route = json.load(open(sys.argv[1]))
    if len(sys.argv) > 2:
        name = sys.argv[2]
SRC = "muse_bf16_proj/runs/v6_router_stacked.py"
s = open(SRC).read()
cu = open("tools/bf16_v34_stream.cu").read()
cfgsrc = open("tools/bf16_v34_cfg.py").read()
assert '"""' not in cu


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


_REN = [("ALL_INSTANCES", "V34_INSTANCES"), ("SMEM_MAX", "V34_SMEM_MAX"), ("SMEM_MIN", "V34_SMEM_MIN"),
        ("SMS", "V34_SMS"), ("LAYERS", "V34_LAYERS"), ("PLANS", "V34_PLANS"), ("TUNES", "V34_TUNES"), ("NTS", "V34_NTS"),
        ("smem", "_v34_smem"), ("valid", "_v34_valid"), ("instance", "_v34_instance"), ("name_expr", "_v34_name"),
        ("candidates", "_v34_candidates"), ("launch", "_v34_launch")]


def ren(py):
    """v34's Python names get a v34 prefix in the runs file (v6 already has CONFIGS etc.)."""
    for a, b in _REN:
        py = re.sub(r"\b%s\b" % a, b, py)
    return py


cfg_body = cfgsrc.split('"""', 2)[2]  # drop the module docstring
if DIAG:
    inst_expr = "V34_INSTANCES"
else:
    inst_expr = "sorted({_v34_instance(k, nt, p, t) for (k, b), (p, t) in V34_ROUTE.items() for nt in _v34_nts(b)})"

HOST = r'''

# ---------------- v34: TMA-bulk weight-streaming CUDA kernel (tools/bf16_v34_stream.cu, tools/bf16_v34_cfg.py) -------
_V34_SRC = r"""
''' + cu + r'''"""
# tools/bf16_v34_cfg.py (body; names prefixed _v34 / V34_)
''' + ren(cfg_body) + ren(r'''

_V34 = {}  # "cubin", "lowered": {inst: name}; device -> (module, {inst: function})


def _v34_nts(bucket):
    return (1,) if bucket <= 8 else ((2,) if bucket <= 16 else (4,))


def _v34_insts():
    return ''' + inst_expr + r'''


def _v34_compile():
    """NVRTC (cuda.bindings, in-process) -> one sm_90a cubin with the needed instantiations. Host only."""
    if "cubin" not in _V34:
        from cuda.bindings import nvrtc
        insts = _v34_insts()
        err, prog = nvrtc.nvrtcCreateProgram(_V34_SRC.encode(), b"bf16_v34.cu", 0, [], [])
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
        _V34["cubin"], _V34["lowered"] = cubin, lowered
    return _V34["cubin"]


def _v34_load(dev):
    """CuPy functions on device dev (compile + load in prepare(), never inside graph capture)."""
    if dev not in _V34:
        import cupy as cp
        cubin = _v34_compile()
        with cp.cuda.Device(dev):
            mod = cp.cuda.function.Module()
            mod.load(cubin)
            fns = {}
            drv = cp.cuda.driver
            for inst, low in _V34["lowered"].items():
                f = mod.get_function(low)
                drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8),
                                     SMEM_MAX)
                fns[inst] = f
        _V34[dev] = (mod, fns)
    return _V34[dev][1]


def _v34_pack(w):
    """[n, K] BF16 -> lossless 12-bit A-fragment stream (tools/bf16_v34_stream.cu): [n/16][K/32][768 B] =
    [lane][step][8 sign|mantissa bytes] + [lane][step][u32 exponent nibbles c = E - lo, register j's two at bits
    4j, 16 + 4j]; lo = the 16-exponent window covering most weights; out-of-window weights (escapes) get c = 0 and an
    fix-up entry (they are stored as code 0 with zero sign / mantissa: decoded 2^(lo - 127)): eoff [n/16 + 1]
    (per-rowtile ranges), emeta (row % 16) << 16 | k, ecorr = the weight (FP32, exact)."""
    n, K = w.shape
    dev = w.device
    v = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    E = (v >> 7) & 0xFF
    h = torch.bincount(E.flatten(), minlength=256).to(torch.int64)
    cs = torch.cumsum(h, 0)
    win = cs[15:] - torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), cs[:-16]])
    lo = int(torch.argmax(win))
    c = E - lo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    ab = torch.where(esc, torch.zeros_like(v), ((v >> 8) & 0x80) | (v & 0x7F)).to(torch.uint8)

    def perm(t):  # [n, K] -> [rt, p, lane, s, 8] (same element order as v33's _v33_repack, steps moved inside lanes)
        t = t.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2).permute(0, 3, 5, 2, 4, 6, 1, 7)  # rt, p, s, g, t, e, h, q
        return t.reshape(n // 16, K // 32, 2, 32, 8).permute(0, 1, 3, 2, 4)
    abp = perm(ab).reshape(n // 16, K // 32, 512)
    cp = perm(c).to(torch.int64)  # [rt, p, lane, s, 8]
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], dtype=torch.int64, device=dev)
    word = (cp << sh).sum(-1)  # disjoint bits
    word = torch.where(word >= 1 << 31, word - (1 << 32), word).to(torch.int32)
    nib = word.contiguous().view(torch.uint8).reshape(n // 16, K // 32, 256)
    packed = torch.cat([abp, nib], -1).contiguous()
    rk = esc.nonzero()  # row-major: sorted by row, then k
    rows, ks = rk[:, 0], rk[:, 1]
    cnt = torch.bincount(rows // 16, minlength=n // 16)
    eoff = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(cnt, 0)]).to(torch.int32)
    emeta = (((rows % 16) << 16) | ks).to(torch.int32)
    ecorr = w[rows, ks].float().contiguous()
    if emeta.numel() == 0:
        emeta = torch.zeros(1, dtype=torch.int32, device=dev)
        ecorr = torch.zeros(1, dtype=torch.float32, device=dev)
    cc = (lo << 7) | (lo << 23)
    return packed, cc, eoff.contiguous(), emeta.contiguous(), ecorr


def _v34_repack(w):
    """[n, K] BF16 -> A-fragment order [n/16][K/32][s][lane][8 bf16] (lossless permutation, same bytes)."""
    n, K = w.shape
    t = w.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2)  # rt, h, g, p, t, s, e, q
    return t.permute(0, 3, 5, 2, 4, 6, 1, 7).contiguous()  # rt, p, s, g, t, e, h, q


def _v34_router_rows(hi, lo):
    """[256, K] hi, lo -> [512, K]: rowtile 2c = hi rows 16c.., rowtile 2c+1 = lo rows 16c.. (summed by HL = 2)."""
    n, K = hi.shape
    return torch.stack([hi.view(n // 16, 16, K), lo.view(n // 16, 16, K)], 1).reshape(2 * n, K)


def _v34(x, key, plan, tune, state):
    """out[m, n_out] = x . W^T on the v34 kernel (BF16 out; FP32 for the router)."""
    import cupy as cp
    import numpy as np
    m, K = x.shape
    rows, Kw, hl, f32, n_out = LAYERS[key]
    nt = 1 if m <= 8 else (2 if m <= 16 else 4)
    assert K == Kw and m <= 8 * nt and x.is_contiguous() and x.data_ptr() % 16 == 0
    inst = instance(key, nt, plan, tune)
    grid, threads, shm, wsn = launch(key, plan, tune, nt)
    wname = "unpooler" if key == "unpooler_s" else key
    wr = state["v34_" + wname]
    cc, eoff, emeta, ecorr = state["v34_esc_" + wname] if not f32 else (0, wr, wr, wr)
    ws, cnt = state["v34_ws"], state["v34_cnt"]
    assert wsn <= ws.numel()
    out = torch.empty((m, n_out), device=x.device, dtype=torch.float32 if f32 else torch.bfloat16)
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    f = _v34_load(dev)[inst]
    c0 = _v34_cnt_off()[(key, tuple(plan))]
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
        f((grid,), (threads,), (np.uint64(wr.data_ptr()), np.uint64(x.data_ptr()), np.int32(K),
                                np.uint64(out.data_ptr()), np.int32(m), np.int32(n_out), np.int32(K),
                                np.int32(plan[1]), np.int32(plan[0]), np.uint64(ws.data_ptr()),
                                np.uint64(cnt.data_ptr() + 4 * c0), np.uint32(cc), np.uint64(eoff.data_ptr()),
                                np.uint64(emeta.data_ptr()), np.uint64(ecorr.data_ptr())), shared_mem=shm)
    return out


def _v34_cnt_off():
    """Launch-counter slots: one run of grid-many u32 per (layer key, plan) (a grid's counters move in lockstep)."""
    if "cnt_off" not in _V34:
        off, o = {}, 0
        for key, plans in PLANS.items():
            rows = LAYERS[key][0]
            for ks, G in plans:
                off[(key, (ks, G))] = o
                o += rows // 16 // G * ks
        _V34["cnt_off"], _V34["cnt_n"] = off, o
    return _V34["cnt_off"]


def _v34_bucket(m):
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))

''')

PREP_ADD = '''    if torch.cuda.is_available() and weights["qkv_gate"].is_cuda:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            pk, cc, eoff, emeta, ecorr = _v34_pack(weights[name].contiguous())
            state["v34_" + name] = pk
            state["v34_esc_" + name] = (cc, eoff, emeta, ecorr)
        state["v34_router"] = _v34_repack(_v34_router_rows(hi, lo))
        _v34_cnt_off()
        dev = weights["qkv_gate"].device
        state["v34_cnt"] = torch.zeros(_V34["cnt_n"], dtype=torch.int32, device=dev)  # launch counters, start at 0
        state["v34_ws"] = torch.empty(V34_SMS * 18 * 512, dtype=torch.float32, device=dev)  # split-K partials
        _v34_load(weights["qkv_gate"].device.index or 0)  # NVRTC compile + module load now, not in capture
'''

s = rep(s, "\ndef prepare(weights):", HOST + "\ndef prepare(weights):")
s = rep(s, '''    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    return state''', '''    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
''' + PREP_ADD + ('''    try:
        _diag_bench(state)  # v34d: diagnostic only, untimed
    except Exception as e:  # never fail the run
        print("v34d diag failed:", repr(e)[:600], flush=True)
''' if DIAG else "") + '''    return state''')

if DIAG:
    BENCH = r'''

_DIAG = {}


def _diag_bench(state):
    """One-time untimed sweep: per (layer, m) the v6 path (kernel_fn as checked), SGLang's cuBLAS path and every
    v34 candidate, each captured in a CUDA graph and replayed with a cold L2 (a 256 MB buffer read first), median of
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
    print(f"v34d diag: {torch.cuda.get_device_name()} SMs {torch.cuda.get_device_properties(dev).multi_processor_count}"
          f" floor(1 tiny kernel) {floor:.2f} us", flush=True)
    try:
        from cuda.bindings import driver as cu
        fns = _v34_load(dev.index or 0)
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
                    out = _v34(x, key, plan, tune, state).float()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                    t = timed(lambda: _v34(x, key, plan, tune, state))
                except Exception as e:
                    print(f"{key} m {m} {plan} {tune} FAILED {repr(e)[:120]}", flush=True)
                    continue
                res.append((t, plan, tune, err))
            res.sort()
            line = [f"{key} m {m}: v6 {t_v6:.2f} base {t_cb:.2f} | v34 " + "; ".join(
                f"{p}{tu} {t:.2f} ({wbytes / max(t - floor, 0.1) / 1e6:.2f} TB/s) e{err:.0e}" for t, p, tu, err in res[:3])]
            if res:
                best[(key, m)] = (res[0], t_v6)
            print("\n".join(line), flush=True)
    print("v34d summary (key m: best v34 / v6 us):", flush=True)
    for (key, m), ((t, p, tu, err), t6) in best.items():
        print(f"  {key}:{m} {t:.2f} / {t6:.2f} {list(p)} {list(tu)} e{err:.0e}", flush=True)
    print(f"v34d diag: done in {time.time() - t0:.0f} s", flush=True)
'''
    s = s + ren(BENCH)
    hdr = '''# okc: problem=muse_bf16_proj
"""v34d (DIAGNOSTIC, not for the board): v6's kernel_fn unchanged, plus the v34 TMA-bulk weight-streaming CUDA
kernel (tools/bf16_v34_stream.cu) compiled in prepare() and an untimed one-time sweep of its candidates against v6
and cuBLAS for every BF16 layer x m <= 32 and the router (tools/build_bf16_v34.py --diag).

v6:'''
    dst = f"muse_bf16_proj/runs/{DIAG_NAME}.py"
else:
    ROUTE_SRC = "V34_ROUTE = {\n" + "".join(
        f"    ({k.split(':')[0]!r}, {int(k.split(':')[1])}): ({tuple(v[0])}, {tuple(v[1])}),\n"
        for k, v in sorted(route.items())) + "}\n"
    s = rep(s, "\n# ---------------- v34:", "\n# (layer key, m bucket 1 / 8 / 16 / 32) -> v34 ((KS, G), (NW, ST, SPS, RT)), from the v34d diagnostic\n"
            + ROUTE_SRC + "\n# ---------------- v34:")
    s = rep(s, '''    m, k = x.shape
    if layer == "router" and m > TRITON_MAX_M:''', '''    m, k = x.shape
    if m <= 32 and x.is_cuda:
        key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
        r31 = V34_ROUTE.get((key, _v34_bucket(m)))
        if r31 is not None:
            return _v34(x.contiguous(), key, r31[0], r31[1], state)
    if layer == "router" and m > TRITON_MAX_M:''')
    hdr = '''# okc: problem=muse_bf16_proj
"""v34: v6 plus a TMA-bulk weight-streaming CUDA kernel (tools/bf16_v34_stream.cu: W repacked into mma.sync
A-fragment order and streamed by a producer warp with cp.async.bulk into a shared-memory ring, x's k-slice staged in
shared memory once per CTA, one CTA per SM on a balanced one-wave grid, split-K reduced in shared memory and then
through a global workspace with per-CTA launch counters (release/acquire, single writer: no atomics)) on the (layer, m) pairs in V34_ROUTE, picked from the v34d
diagnostic's per-shape timings vs v6 / cuBLAS. Everything else is v6. tools/build_bf16_v34.py.

v6:'''
    dst = f"muse_bf16_proj/runs/{name}.py"
s = rep(s, '# okc: problem=muse_bf16_proj\n"""', hdr + " ")
open(dst, "w").write(s)
print("wrote", dst)
