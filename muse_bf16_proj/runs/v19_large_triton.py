# okc: problem=muse_bf16_proj
"""v19: v7 plus Triton on o / pooler / router at m 128 / 256 (verify sizes), where v7 fell
back to cuBLAS at about 1.00x. Small-m path unchanged. Split-K workspace allocated in prepare().
LARGE_CFG keys are exact scored m (exploration); a follow-up will re-bucket by regime from the
per-shape table. qkv_gate and unpooler stay on SGLang."""

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
# Exact scored m > 32 (exploration). Absent -> SGLang. Re-bucket by regime after this check.
LARGE_CFG = {
    ("o", 128): (64, 64, 1, 4, 3),
    ("o", 256): (64, 64, 1, 8, 3),
    ("pooler", 128): (64, 128, 1, 4, 3),
    ("pooler", 256): (64, 128, 1, 8, 3),
    ("router", 128): (32, 256, 4, 4, 4),
    ("router", 256): (32, 256, 4, 8, 3),
}
TRITON_MAX_M = 32


def _bm(m):
    return 16 if m <= 16 else (32 if m <= 32 else (64 if m <= 64 else 128))


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    if router.is_cuda:
        state["ws"] = torch.empty((256, 256), device=router.device, dtype=torch.float32)
    return state


def _baseline(x, layer, n, state):
    """SGLang's own path for the BF16 layers (as in baseline.py), kept where it is faster.
    (The router never comes here: Triton up to 32 tokens, _router_stacked above.)"""
    if x.device.type == "cpu":  # local CPU testing only (no SGLang / cuBLAS bf16 path)
        return x @ state[layer][:n].T
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)


def _run_triton(x, w, lo, m, n, k, bn, bk, split, warps, stages, has_lo, state):
    bm = _bm(m)
    x = x.contiguous()
    if split > 1:
        if "ws" in state and m <= state["ws"].shape[0] and n <= state["ws"].shape[1]:
            out = state["ws"][:m, :n]
            out.zero_()
        else:
            out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
        out_bf16 = False
    else:
        out = torch.empty((m, n), device=x.device, dtype=torch.float32 if has_lo else x.dtype)
        out_bf16 = not has_lo
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm), split)
    _swapab_gemm[grid](x, w, lo, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                       BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_LO=has_lo,
                       OUT_BF16=out_bf16, num_warps=warps, num_stages=stages)
    return out


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m, k = x.shape
    if m > TRITON_MAX_M and (layer, m) in LARGE_CFG:
        bn, bk, split, warps, stages = LARGE_CFG[(layer, m)]
        if layer == "router":
            return _run_triton(x, state["router_hi"], state["router_lo"], m, n, k,
                               bn, bk, split, warps, stages, True, state)
        return _run_triton(x, state[layer], state[layer], m, n, k,
                           bn, bk, split, warps, stages, False, state)
    if layer == "router" and m > TRITON_MAX_M:
        return _router_stacked(x, state)
    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
    key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
    if CONFIGS[key] is None:
        return _baseline(x, layer, n, state)
    bn, bk, split, warps, stages = CONFIGS[key]
    if layer == "router":
        return _run_triton(x, state["router_hi"], state["router_lo"], m, n, k,
                           bn, bk, split, warps, stages, True, state)
    return _run_triton(x, state[layer], state[layer], m, n, k,
                       bn, bk, split, warps, stages, False, state)


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
