# okc: problem=muse_bf16_proj
"""Dense projections: a swap-AB Triton GEMM for small m (weights as the MMA's M side, tokens
padded to 16 as its N side), with split-K + FP32 atomics for the narrow router.

- BF16 layers: out^T[n, m] = W[n, :] . x[m, :], FP32 accumulation, BF16 store. Weights untouched.
- Router: FP32 weights as a BF16 hi + lo pair (allowed by the rules at any m), both dotted
  against the same x tile in one pass, FP32 accumulation, split-K with FP32 atomic adds.
- Where cuBLAS won in H200 check runs (qkv_gate, unpooler, and every layer above 32 tokens)
  SGLang's own path is kept."""

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
                 HAS_LO: tl.constexpr, OUT_BF16: tl.constexpr, PARTIAL: tl.constexpr = False):
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
    if PARTIAL:  # split-K partials to [SPLIT_K, M, N] FP32 (no memset, no atomics); _reduce_splits sums
        tl.store(o_ptrs + pid_k * M * stride_om, acc, mask=o_mask)
    elif SPLIT_K > 1:
        tl.atomic_add(o_ptrs, acc, mask=o_mask, sem="relaxed")
    elif OUT_BF16:
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)
    else:
        tl.store(o_ptrs, acc, mask=o_mask)


@triton.jit
def _reduce_splits(p_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    """out[i] = bf16(sum_s p[s, i]), FP32 sum in a fixed order (deterministic)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for sidx in tl.static_range(SPLIT_K):
        acc += tl.load(p_ptr + sidx * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


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
# Sweep C (check only, exact m to learn 4 configs per layer in one run): split-K with FP32
# partials + one reduce/cast kernel. (BN, BK, SPLIT_K, warps, stages)
PARTIAL_SWEEP = {
    ("qkv_gate", 1): (64, 128, 8, 4, 4), ("qkv_gate", 8): (64, 128, 16, 4, 4),
    ("qkv_gate", 16): (32, 128, 8, 4, 4), ("qkv_gate", 32): (64, 128, 8, 8, 4),
    ("pooler", 1): (64, 128, 4, 4, 4), ("pooler", 8): (64, 128, 8, 4, 4),
    ("pooler", 16): (32, 128, 4, 4, 4), ("pooler", 32): (64, 128, 4, 8, 4),
    ("unpooler_s", 1): (32, 128, 8, 4, 4), ("unpooler_s", 8): (32, 128, 16, 4, 4),
    ("unpooler", 16): (64, 128, 2, 4, 4), ("unpooler", 32): (64, 128, 4, 4, 4),
    ("o", 1): (64, 128, 2, 4, 4), ("o", 8): (64, 64, 4, 4, 4),
    ("o", 16): (32, 128, 2, 4, 4), ("o", 32): (64, 128, 2, 8, 4),
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
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    return state


def _baseline(x, layer, n, state):
    """SGLang's own path (as in baseline.py), kept where it is faster."""
    if layer != "router":
        try:
            from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
        except Exception:  # local CPU testing only
            return x @ state[layer][:n].T
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.device.type == "cpu":  # local CPU testing only
        return x.float() @ (state["router_hi"].float() + state["router_lo"].float()).T
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


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
    if (key, m) in PARTIAL_SWEEP:
        return _partial(x, state[layer], n, *PARTIAL_SWEEP[(key, m)])
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


def _partial(x, w, n, bn, bk, split, warps, stages):
    x = x.contiguous()
    m, k = x.shape
    bm = _bm(m)
    part = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    _swapab_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm), split)](
        x, w, w, part, m, n, k, x.stride(0), w.stride(0), n, BM=bm, BN=bn, BK=bk, SPLIT_K=split,
        HAS_LO=False, OUT_BF16=False, PARTIAL=True, num_warps=warps, num_stages=stages)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _reduce_splits[(triton.cdiv(m * n, 1024),)](part, out, m * n, SPLIT_K=split, BLOCK=1024, num_warps=4)
    return out


def _router_stacked(x, state):
    """Router above 32 tokens: SGLang's hi + lo pair as ONE cuBLAS GEMM against the stacked
    [hi; lo] weights (FP32 out), then hi-part + lo-part: the same two FP32-accumulated products
    and the same final FP32 add as SGLang's two torch.mm calls, in one GEMM launch."""
    if x.device.type == "cpu":  # local CPU testing only
        both = x.float() @ state["router_hilo_t"].float()
    else:
        both = torch.mm(x, state["router_hilo_t"], out_dtype=torch.float32)
    n = both.shape[1] // 2
    return both[:, :n] + both[:, n:]
