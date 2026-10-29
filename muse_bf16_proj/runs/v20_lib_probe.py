# okc: problem=muse_bf16_proj
"""v20: v7 plus an in-prepare probe of judge-image GEMM libraries (flashinfer / quack /
tilelang / cutlass.cute / deep_gemm / sgl_kernel / cuda.tile) and, when one exposes a usable
BF16 GEMM, a cold-L2 microbench on qkv_gate at m in {1,8,16,32,128,256} against SGLang.
kernel_fn is identical to v7 (no new route yet). Check-only: prints PROBE/BENCH lines for the
next routing decision. Caps prepare at ~60s."""

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
    if weights["qkv_gate"].is_cuda:
        _probe_and_bench(state)
    return state


def _probe_and_bench(state):
    """Discover GEMM libs on the judge and, if any look usable, time them on qkv_gate."""
    import importlib, time, traceback
    found = {}
    for name in ("flashinfer", "flashinfer.gemm", "quack", "tilelang", "cutlass", "cutlass.cute",
                 "deep_gemm", "sgl_kernel", "cuda.tile", "cuda.bindings.nvrtc"):
        try:
            m = importlib.import_module(name)
            attrs = [a for a in dir(m) if not a.startswith("_") and any(k in a.lower() for k in
                     ("gemm", "mm", "matmul", "bf16", "linear", "cutedsl", "tgv", "grouped"))]
            found[name] = attrs[:30]
            print(f"PROBE {name} OK attrs={attrs[:30]}", flush=True)
        except Exception as ex:
            print(f"PROBE {name} FAIL {type(ex).__name__}: {str(ex)[:120]}", flush=True)
    # Try known call shapes against qkv_gate.
    W = state["qkv_gate"]
    n, k = W.shape
    candidates = []
    # flashinfer mm_fp8 / gemm if present
    try:
        import flashinfer
        for attr in ("gemm", "mm_fp16", "bmm_fp16", "single_prefill_with_kv_cache"):
            if hasattr(flashinfer, attr):
                print(f"PROBE flashinfer.{attr}={getattr(flashinfer, attr)}", flush=True)
        if hasattr(flashinfer, "gemm"):
            candidates.append(("flashinfer.gemm", flashinfer.gemm))
    except Exception as ex:
        print(f"PROBE flashinfer detail FAIL {ex}", flush=True)
    try:
        from sglang.kernels.ops.gemm import cutedsl_bf16_gemm
        candidates.append(("sglang.cutedsl_bf16_gemm", cutedsl_bf16_gemm))
        print("PROBE sglang.cutedsl_bf16_gemm OK", flush=True)
    except Exception as ex:
        print(f"PROBE sglang.cutedsl_bf16_gemm FAIL {type(ex).__name__}: {str(ex)[:160]}", flush=True)
    try:
        import quack
        for a in dir(quack):
            if "gemm" in a.lower() or "mm" == a.lower():
                print(f"PROBE quack.{a}", flush=True)
                candidates.append((f"quack.{a}", getattr(quack, a)))
    except Exception as ex:
        print(f"PROBE quack detail FAIL {ex}", flush=True)
    # Microbench: each candidate vs SGLang on qkv at several m. Flush L2 between runs with a big memset.
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    trash = torch.empty((64 << 20,), device=W.device, dtype=torch.uint8)  # 64 MB
    def cold():
        trash.fill_(0)
        torch.cuda.synchronize()
    def time_fn(fn, xs, reps=20):
        for x in xs: fn(x)  # warmup
        torch.cuda.synchronize()
        ts = []
        for x in xs:
            cold()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6 / reps)
        return ts
    ms = [1, 8, 16, 32, 128, 256]
    xs = [torch.randn((m, k), device=W.device, dtype=torch.bfloat16) for m in ms]
    def sgl(x):
        return _bf16_gemm_dispatch_impl(x, W, None)
    try:
        t_sgl = time_fn(sgl, xs)
        print(f"BENCH sglang ms={ms} us={[round(t,2) for t in t_sgl]}", flush=True)
    except Exception as ex:
        print(f"BENCH sglang FAIL {ex}", flush=True)
        return
    for label, fn in candidates:
        try:
            def run(x, _fn=fn, _W=W):
                try:
                    return _fn(x, _W)
                except TypeError:
                    return _fn(x, _W.t())  # some want [K,N]
            t = time_fn(run, xs)
            # correctness at m=16
            y = run(xs[2]); ref = sgl(xs[2])
            err = (y.float() - ref.float()).norm() / ref.float().norm()
            print(f"BENCH {label} ms={ms} us={[round(u,2) for u in t]} relL2={float(err):.3e}", flush=True)
        except Exception as ex:
            print(f"BENCH {label} FAIL {type(ex).__name__}: {str(ex)[:200]}", flush=True)
            traceback.print_exc()



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
