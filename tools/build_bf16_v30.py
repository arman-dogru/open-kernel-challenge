"""Build the bf16 v30 runs files from runs/v6_router_stacked.py (board 1.115x) + tools/bf16_v30_stream.cu +
tools/bf16_v30_cfg.py. Run from the repo root.

  python3 tools/build_bf16_v30.py --diag   -> muse_bf16_proj/runs/v30d_bench.py  (DIAGNOSTIC: v6's kernel_fn, untimed
                                              in-prepare timing of every v30 config vs v6 / cuBLAS per layer x m)
  python3 tools/build_bf16_v30.py ROUTE.json -> muse_bf16_proj/runs/v30_routed.py (v6 + the v30 kernel on the
                                              (layer, m-range) pairs in ROUTE.json; everything else v6)
"""
import json, sys

DIAG = "--diag" in sys.argv
route = {}
if not DIAG:
    route = json.load(open(sys.argv[1]))
SRC = "muse_bf16_proj/runs/v6_router_stacked.py"
s = open(SRC).read()
cu = open("tools/bf16_v30_stream.cu").read()
cfgsrc = open("tools/bf16_v30_cfg.py").read()
assert '"""' not in cu


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


cfg_body = cfgsrc.split('"""', 2)[2]  # drop the module docstring
import re
_REN = [("ALL_INSTANCES", "V30_INSTANCES"), ("CONFIGS", "V30_CONFIGS"), ("NTS", "V30_NTS"), ("LAYERS", "V30_LAYERS"),
        ("name_expr", "_v30_name"), ("valid", "_v30_valid"), ("smem", "_v30_smem"), ("grid", "_v30_grid")]


def ren(py):
    """v30's Python names get a v30 prefix in the runs file (v6 already has CONFIGS)."""
    for a, b in _REN:
        py = re.sub(r"\b%s\b" % a, b, py)
    return py

HOST = r'''

# ---------------- v30: weight-streaming CUDA kernel (tools/bf16_v30_stream.cu, tools/bf16_v30_cfg.py) ----------------
_V30_SRC = r"""
''' + cu + r'''"""
# tools/bf16_v30_cfg.py (body; names prefixed _v30 / V30_)
''' + ren(cfg_body) + ren(r'''

_V30 = {}  # "cubin", "lowered": {inst: name}; device -> {inst: function}


def _v30_compile():
    """NVRTC (cuda.bindings, in-process) -> one sm_90a cubin with every instantiation. Host only."""
    if "cubin" not in _V30:
        from cuda.bindings import nvrtc
        err, prog = nvrtc.nvrtcCreateProgram(_V30_SRC.encode(), b"bf16_stream.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        for inst in ALL_INSTANCES:
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
        for inst in ALL_INSTANCES:
            _, low = nvrtc.nvrtcGetLoweredName(prog, name_expr(inst).encode())
            lowered[inst] = low.decode()
        _V30["cubin"], _V30["lowered"] = cubin, lowered
    return _V30["cubin"]


def _v30_load(dev):
    """CuPy functions on device dev (compile + load in prepare(), never inside graph capture)."""
    if dev not in _V30:
        import cupy as cp
        cubin = _v30_compile()
        with cp.cuda.Device(dev):
            mod = cp.cuda.function.Module()
            mod.load(cubin)
            fns = {}
            drv = cp.cuda.driver
            for inst, low in _V30["lowered"].items():
                f = mod.get_function(low)
                rt, nt, wk, rg, cs, gp = inst
                drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8),
                                     smem((rt, wk, rg, cs, gp), nt))
                fns[inst] = f
        _V30[dev] = (mod, fns)
    return _V30[dev][1]


def _v30_repack(w):
    """[n, K] BF16 -> A-fragment order [n/16][K/32][s][lane][8 bf16] (lossless permutation, same bytes)."""
    n, K = w.shape
    t = w.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2)  # rt, h, g, p, t, s, e, q
    return t.permute(0, 3, 5, 2, 4, 6, 1, 7).contiguous()  # rt, p, s, g, t, e, h, q


def _v30(x, key, n, cfg, state):
    """out[m, n] (BF16) = x . W[:n]^T on the v30 kernel with cfg = (RT, WK, RG, CS, GP)."""
    import cupy as cp
    import numpy as np
    m, K = x.shape
    nt = (m + 7) // 8
    rt, wk, rg, cs, gp = cfg
    wr = state["v30_" + ("unpooler" if key == "unpooler_s" else key)]
    assert x.is_contiguous() and x.data_ptr() % 16 == 0 and K % 8 == 0
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    nblk, units = grid(cfg, n)
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    f = _v30_load(dev)[(rt, nt, wk, rg, cs, gp)]
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
        f((nblk,), (32 * wk * rg,), (np.uint64(wr.data_ptr()), np.uint64(x.data_ptr()), np.int32(K),
                                     np.uint64(out.data_ptr()), np.int32(m), np.int32(n), np.int32(K),
                                     np.int32(units)), shared_mem=smem(cfg, nt))
    return out


def _v30_bucket(m):
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))

''')

PREP_ADD = '''    if torch.cuda.is_available() and weights["qkv_gate"].is_cuda:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            state["v30_" + name] = _v30_repack(weights[name].contiguous())
        _v30_load(weights["qkv_gate"].device.index or 0)  # NVRTC compile + module load now, not in capture
'''

s = rep(s, "\ndef prepare(weights):", HOST + "\ndef prepare(weights):")
s = rep(s, '''    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    return state''', '''    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
''' + PREP_ADD + ('''    try:
        _diag_bench(state)  # v30d: diagnostic only, untimed
    except Exception as e:  # never fail the run
        print("v30d diag failed:", repr(e)[:600], flush=True)
''' if DIAG else "") + '''    return state''')

if DIAG:
    BENCH = r'''

_DIAG = {}


def _diag_bench(state):
    """One-time untimed sweep: per (layer, m) the v6 path (kernel_fn as checked), SGLang's cuBLAS path and every
    valid v30 config, each captured in a CUDA graph and replayed with a cold L2 (a 256 MB buffer read first), median
    of 15. Prints us and the v30 output's max |diff| vs cuBLAS (relative to max |ref|)."""
    import time
    if _DIAG or not torch.cuda.is_available():
        return
    _DIAG["done"] = True
    t0 = time.time()
    dev = state["qkv_gate"].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
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

    print("v30d diag:", torch.cuda.get_device_name(), flush=True)
    best = {}
    for key, (n, K) in LAYERS.items():
        layer = "unpooler" if key == "unpooler_s" else key
        for m in ((1, 8) if key == "unpooler_s" else (1, 8, 16, 32)):
            if time.time() - t0 > 200:
                print("(time guard)", flush=True)
                break
            x = torch.randn((m, K), generator=g, device=dev).to(torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}
            ref = _baseline(x, layer, n, state).float()
            t_v6 = timed(lambda: kernel_fn(data, state))
            t_cb = timed(lambda: _baseline(x, layer, n, state))
            line = [f"{key} m {m}: v6 {t_v6:.2f} cuBLAS {t_cb:.2f}"]
            res = []
            for cfg in CONFIGS:
                if not valid(cfg, n, K, (m + 7) // 8):
                    continue
                try:
                    out = _v30(x, key, n, cfg, state).float()
                    torch.cuda.synchronize()
                    err = ((out - ref).abs().max() / ref.abs().max()).item()
                    t = timed(lambda: _v30(x, key, n, cfg, state))
                except Exception as e:
                    line.append(f"{cfg} FAILED {repr(e)[:80]}")
                    continue
                res.append((t, cfg, err))
            res.sort()
            for t, cfg, err in res[:6]:
                line.append(f"  {cfg} {t:.2f} err {err:.1e}")
            if res:
                best[(key, m)] = res[0]
            print("\n".join(line), flush=True)
    print("v30d summary (key m: v30 best / v6):", flush=True)
    for (key, m), (t, cfg, err) in best.items():
        print(f"  {key} {m}: {t:.2f} {cfg} err {err:.1e}", flush=True)
    print(f"v30d diag: done in {time.time() - t0:.0f} s", flush=True)
'''
    s = s + ren(BENCH)
    hdr = '''# okc: problem=muse_bf16_proj
"""v30d (DIAGNOSTIC, not for the board): v6's kernel_fn unchanged, plus the v30 weight-streaming CUDA kernel
(tools/bf16_v30_stream.cu) compiled in prepare() and an untimed one-time sweep of its configs against v6 and cuBLAS
for every BF16 layer x m <= 32 (tools/build_bf16_v30.py --diag).

v6:'''
    dst = "muse_bf16_proj/runs/v30d_bench.py"
else:
    ROUTE_SRC = "V30_ROUTE = {\n" + "".join(f"    ({k.split(':')[0]!r}, {int(k.split(':')[1])}): {tuple(v)},\n"
                                           for k, v in sorted(route.items())) + "}\n"
    s = rep(s, "\ndef prepare(weights):", "\n# (layer key, m bucket 1 / 8 / 16 / 32) -> v30 config, from the v30d diagnostic\n" + ROUTE_SRC + "\ndef prepare(weights):")
    s = rep(s, '''    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
    key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer''', '''    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
    key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
    v30cfg = V30_ROUTE.get((key, _v30_bucket(m)))
    if v30cfg is not None and x.is_cuda:
        return _v30(x.contiguous(), key, n, v30cfg, state)''')
    hdr = '''# okc: problem=muse_bf16_proj
"""v30: v6 plus a weight-streaming CUDA kernel (tools/bf16_v30_stream.cu: W repacked into mma.sync A-fragment
order, 512-byte LDG.128 streams with bulk L2 prefetch, split-K reduced in shared memory and across a thread-block
cluster through DSMEM, no atomics, persistent grid) on the (layer, m) pairs in V30_ROUTE, picked from the v30d
diagnostic's per-shape timings vs v6 / cuBLAS. Router unchanged (v6's stacked hi/lo). tools/build_bf16_v30.py.

v6:'''
    dst = "muse_bf16_proj/runs/v30_routed.py"
s = rep(s, '# okc: problem=muse_bf16_proj\n"""', hdr + " ")
open(dst, "w").write(s)
print("wrote", dst)
