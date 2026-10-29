# okc: problem=muse_bf16_proj
"""v12 (check-only exploration): v11's bench at m = 128 / 256 too, plus a print-only probe of which
in-process JIT toolchains the judge image has (no processes are started).
v11: v7 unchanged for kernel_fn; prepare() additionally times cuBLAS
layout/API variants (pre-transposed weights, gemv, swapped operands) and v7's Triton configs against
SGLang's dispatch, per layer and token count, in CUDA graphs with a cold L2, and prints the results
(BENCH lines). Nothing measured is kept or used by kernel_fn. A submit sets BENCH = False.

v7: Dense projections: a swap-AB Triton GEMM for small m (weights as the MMA's M side, tokens
padded to 16 as its N side), with split-K + FP32 atomics for the narrow router.

- BF16 layers: out^T[n, m] = W[n, :] . x[m, :], FP32 accumulation, BF16 store. Weights untouched.
- Router: FP32 weights as a BF16 hi + lo pair (allowed by the rules at any m), both dotted
  against the same x tile in one pass, FP32 accumulation, split-K with FP32 atomic adds
  (order-nondeterministic in the last FP32 bits, ~1e-7 relative; allowed by the rules; a
  deterministic partials + reduce variant costs an extra launch, see sweep C).
- Router above 32 tokens: the same hi + lo products as SGLang's two cuBLAS GEMMs, run as one
  cuBLAS GEMM against the stacked [hi; lo] weights (FP32 out), then hi-part + lo-part in FP32.
- Where cuBLAS won in H200 check runs (qkv_gate, unpooler, and every other layer above 32
  tokens) SGLang's own path is kept."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _swapab_gemm(x_ptr, w_ptr, lo_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr,
                 HAS_LO: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    k_per = K // SPLIT_K
    k0 = pid_k * k_per
    n_mask = offs_n < N
    m_mask = offs_m < M
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + (k0 + offs_k)[:, None]
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        xt = tl.load(x_ptrs + kk, mask=m_mask[None, :], other=0.0)
        w = tl.load(w_ptrs + kk, mask=n_mask[:, None], other=0.0)
        acc = tl.dot(w, xt, acc)
        if HAS_LO:
            lo = tl.load(lo_ptr + offs_n[:, None] * stride_wn + (k0 + kk + offs_k)[None, :],
                         mask=n_mask[:, None], other=0.0)
            acc = tl.dot(lo, xt, acc)
    o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    o_mask = m_mask[None, :] & n_mask[:, None]
    if SPLIT_K > 1:
        tl.atomic_add(o_ptrs, acc, mask=o_mask, sem="relaxed")
    elif OUT_BF16:
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)
    else:
        tl.store(o_ptrs, acc, mask=o_mask)


# (BN, BK, SPLIT_K, num_warps, num_stages) per layer, for m <= 32, picked per shape from H200
# check runs (v1/v2 in runs/). Layers set to None keep SGLang's cuBLAS path, which won there.
CONFIGS = {
    "qkv_gate": None,
    "o": (64, 128, 1, 4, 4),
    "pooler": (32, 256, 1, 4, 4),
    "unpooler_s": None,
    "unpooler": None,
    "router": (16, 256, 8, 4, 4),
}
# Above this m, cuBLAS (SGLang's path) wins on every layer so far.
TRITON_MAX_M = 32


def _bm(m):
    return 16 if m <= 16 else (32 if m <= 32 else 64)


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    if BENCH and weights["qkv_gate"].is_cuda:
        _bench(state)
    return state


def _baseline(x, layer, n, state):
    """SGLang's own path for the BF16 layers (as in baseline.py), kept where it is faster.
    (The router never comes here: Triton up to 32 tokens, _router_stacked above.)"""
    if x.device.type == "cpu":  # local CPU testing only (no SGLang / cuBLAS bf16 path)
        return x @ state[layer][:n].T
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m, k = x.shape
    if layer == "router" and m > TRITON_MAX_M:
        return _router_stacked(x, state)
    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
    key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
    if CONFIGS[key] is None:
        return _baseline(x, layer, n, state)
    bn, bk, split, warps, stages = CONFIGS[key]
    bm = _bm(m)
    if layer == "router":
        w, lo = state["router_hi"], state["router_lo"]
    else:
        w, lo = state[layer], state[layer]
    x = x.contiguous()
    if split > 1:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm), split)
    _swapab_gemm[grid](x, w, lo, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                       BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_LO=(layer == "router"),
                       OUT_BF16=(split == 1), num_warps=warps, num_stages=stages)
    if layer != "router" and split > 1:
        out = out.to(x.dtype)
    return out


def _router_stacked(x, state):
    """Router above 32 tokens: SGLang's hi + lo pair as ONE cuBLAS GEMM against the stacked
    [hi; lo] weights (FP32 out), then hi-part + lo-part: the same precision as SGLang's two
    torch.mm calls (BF16 inputs, FP32 accumulation, FP32 add), in one GEMM launch
    (1.56-1.60x at 128 / 256 tokens in check c3f668686d154849)."""
    if x.device.type == "cpu":  # local CPU testing only
        both = x.float() @ state["router_hilo_t"].float()
    else:
        both = torch.mm(x, state["router_hilo_t"], out_dtype=torch.float32)
    n = both.shape[1] // 2
    return both[:, :n] + both[:, n:]


# ------------------------------- exploration bench (check runs only) -------------------------------
BENCH = True
_BENCHED = []


def _time(fn, flush, reps=40):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    ts = []
    for _ in range(reps):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], out


def _tri(x, w, n, cfg):
    bn, bk, split, warps, stages = cfg
    m, k = x.shape
    bm = _bm(m)
    if split > 1:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _swapab_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm), split)](
        x, w, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0), BM=bm, BN=bn, BK=bk, SPLIT_K=split,
        HAS_LO=False, OUT_BF16=(split == 1), num_warps=warps, num_stages=stages)
    return out.to(x.dtype) if split > 1 else out


def _bench(state):
    if _BENCHED:
        return
    _BENCHED.append(1)
    _probe()
    import time
    t0 = time.time()
    dev = state["qkv_gate"].device
    flush = torch.empty((512 * 1024 * 1024 // 4,), device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(5)
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl as disp
    tri_cfgs = [(64, 128, 1, 4, 4), (32, 256, 1, 4, 4), (16, 256, 4, 4, 4), (32, 256, 2, 4, 4), (64, 256, 2, 8, 3)]
    for layer, n_full in (("qkv_gate", 2304), ("o", 8192), ("pooler", 4096), ("unpooler", 8192)):
        W = state[layer]
        for m in (1, 8, 16, 32, 128, 256):
            n = 1024 if (layer == "unpooler" and m <= 8) else n_full
            Wn = W[:n]
            Wt = Wn.t().contiguous()
            x = torch.randn((m, W.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            tb, ref = _time(lambda: disp(x, Wn, None), flush)
            res = []
            cands = {
                "linear": lambda: torch.nn.functional.linear(x, Wn),
                "mm_Wt": lambda: torch.mm(x, Wt),
                "swap": lambda: torch.mm(Wn, x.t()).t(),
                "swap_c": lambda: torch.mm(Wn, x.t()).t().contiguous(),
                "matmul_Wt": lambda: torch.matmul(x, Wt),
            }
            if m > 32:
                cands = {k: v for k, v in cands.items() if k in ("linear", "mm_Wt", "matmul_Wt")}
            if m == 1:
                cands["mv"] = lambda: torch.mv(Wn, x[0]).unsqueeze(0)
                cands["mvT"] = lambda: torch.mv(Wt.t(), x[0]).unsqueeze(0)
            for i, c in enumerate(tri_cfgs if m <= 32 else []):
                cands[f"tri{c}"] = (lambda c=c: _tri(x, Wn, n, c))
            for name, fn in cands.items():
                try:
                    t, out = _time(fn, flush)
                    err = float((out.float() - ref.float()).norm() / ref.float().norm())
                    res.append((t, name, err))
                except Exception as ex:
                    res.append((1e9, name + " ERR " + type(ex).__name__ + ":" + str(ex)[:60], 0.0))
            res.sort()
            line = " ".join(f"{nm}={t:.2f}({e:.0e})" for t, nm, e in res[:6])
            print(f"BENCH {layer} m={m} n={n} sglang={tb:.2f}us | {line}", flush=True)
    print(f"BENCH done {time.time() - t0:.0f}s", flush=True)


def _probe():
    """Print which in-process compilers / DSLs import in the judge image (diagnostics only)."""
    import importlib, importlib.util, shutil, sys
    print("PROBE python", sys.version.split()[0], "torch", torch.__version__, "triton", triton.__version__, flush=True)
    for mod in ("cupy", "cuda", "cuda.bindings", "cuda.bindings.nvrtc", "cuda.bindings.driver", "cuda.core",
                "cuda.core.experimental", "cuda.nvrtc", "cutlass", "cutlass.cute", "tilelang", "flashinfer",
                "tvm", "nvidia.cuda_nvrtc", "numba.cuda", "deep_gemm", "sgl_kernel", "vllm", "pycuda", "quack",
                "thunderkittens", "cuda.tile", "helion"):
        try:
            spec = importlib.util.find_spec(mod)
        except Exception as ex:
            spec = None
        print("PROBE", mod, "found" if spec is not None else "-", flush=True)
    print("PROBE nvcc", shutil.which("nvcc"), "ptxas", shutil.which("ptxas"), flush=True)
    try:
        from torch.cuda import jiterator  # noqa: F401
        print("PROBE jiterator ok", flush=True)
    except Exception as ex:
        print("PROBE jiterator", type(ex).__name__, flush=True)
