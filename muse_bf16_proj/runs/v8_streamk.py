# okc: problem=muse_bf16_proj
"""v8: stream-K for the big small-m layers (qkv_gate, pooler, unpooler at m <= 32), else as v7.

Stream-K (_sk_gemv): the weight matrix is cut into (row block, BK chunk) units and the units are
dealt out in equal contiguous ranges to one persistent wave of programs (a few per SM), so every
SM streams the same number of weight bytes whatever n and K are (v7's grids for these layers were
72-128 programs on 132 SMs, and cuBLAS adds a separate split-K reduce kernel). A program that
covers a whole row block stores BF16 directly; one that covers part of it adds its FP32 partial
into a FP32 scratch tile with atomics, then bumps the row block's counter; the program that
completes the block reads the FP32 sum, stores BF16 and puts the scratch tile and the counter
back to zero (so the next call, or graph replay, starts clean). BF16 operands, FP32
accumulation, one BF16 rounding at the end: the same precision as v7 / cuBLAS. FP32 atomic adds
are order-nondeterministic in the last FP32 bits (~1e-7 relative), as the router's split-K in v7.

Dense projections (v7): a swap-AB Triton GEMM for small m (weights as the MMA's M side, tokens
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


@triton.jit
def _sk_gemv(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K, stride_xm, stride_wn, stride_om,
             TOTAL, NPROG, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Stream-K small-m GEMM (see the module docstring). out[m, n] = sum_k x[m, k] W[n, k]."""
    pid = tl.program_id(0)
    upr = K // BK                       # units (BK chunks) per row block
    start = (pid * TOTAL) // NPROG
    end = ((pid + 1) * TOTAL) // NPROG
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_b = tl.arange(0, BN)
    m_mask = offs_m < M
    u = start
    while u < end:
        rb = u // upr
        kb0 = u - rb * upr
        kb1 = tl.minimum(upr, kb0 + (end - u))
        offs_n = rb * BN + offs_b
        n_mask = offs_n < N
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
        x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        for kb in range(kb0, kb1):
            xt = tl.load(x_ptrs + kb * BK, mask=m_mask[None, :], other=0.0)
            w = tl.load(w_ptrs + kb * BK, mask=n_mask[:, None], other=0.0)
            acc = tl.dot(w, xt, acc)
        o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
        o_mask = m_mask[None, :] & n_mask[:, None]
        nu = kb1 - kb0
        if nu == upr:
            tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)
        else:
            # scratch tile for this row block: [BN, BM] FP32, row-block major
            ws_ptrs = ws_ptr + rb * (BN * BM) + offs_b[:, None] * BM + offs_m[None, :]
            tl.atomic_add(ws_ptrs, acc, sem="relaxed")
            tl.debug_barrier()
            old = tl.atomic_add(cnt_ptr + rb, nu, sem="acq_rel")
            if old + nu == upr:
                # read the summed tile with an atomic (+0, at L2 where the adds landed), then zero it
                tot = tl.atomic_add(ws_ptrs, tl.zeros((BN, BM), dtype=tl.float32), sem="acquire")
                tl.store(o_ptrs, tot.to(out_ptr.dtype.element_ty), mask=o_mask)
                tl.store(ws_ptrs, tl.zeros((BN, BM), dtype=tl.float32))
                tl.store(cnt_ptr + rb, 0)
        u += nu


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
# Stream-K layers at m <= 32 (BF16 weights of at least SK_MIN_BYTES): (BN, BK, programs per SM,
# num_warps, num_stages). Check-only sweep: one config per token count while exploring.
SK_MIN_BYTES = 8 << 20
SK_SWEEP = {1: (64, 256, 1, 4, 3), 8: (32, 256, 2, 4, 3), 16: (64, 128, 2, 4, 4),
            32: (64, 256, 1, 4, 4)}
SK_CONFIG = (64, 256, 1, 4, 3)
SK_MAX_ROW_BLOCKS = 4096
SK_WS = SK_MAX_ROW_BLOCKS * 64 * 32  # FP32 scratch elements (row blocks x BN x BM upper bound)


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
    dev = router.device
    state["sk_ws"] = torch.zeros(SK_WS, device=dev, dtype=torch.float32)
    state["sk_cnt"] = torch.zeros(SK_MAX_ROW_BLOCKS, device=dev, dtype=torch.int32)
    state["n_sm"] = (torch.cuda.get_device_properties(dev).multi_processor_count
                     if dev.type == "cuda" else 4)
    return state


def _streamk(x, w, n, state, cfg):
    bn, bk, per_sm, warps, stages = cfg
    m, k = x.shape
    bm = _bm(m)
    row_blocks = triton.cdiv(n, bn)
    total = row_blocks * (k // bk)
    nprog = min(total, state["n_sm"] * per_sm)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _sk_gemv[(nprog,)](x, w, out, state["sk_ws"], state["sk_cnt"], m, n, k, x.stride(0),
                       w.stride(0), out.stride(0), total, nprog, BM=bm, BN=bn, BK=bk,
                       num_warps=warps, num_stages=stages)
    return out


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
    w_full = state[layer]
    if (layer != "router" and n * k * 2 >= SK_MIN_BYTES and m <= 32
            and k % 256 == 0 and triton.cdiv(n, 32) <= SK_MAX_ROW_BLOCKS):
        cfg = SK_SWEEP.get(m, SK_CONFIG)
        if k % cfg[1] == 0 and cfg[0] * _bm(m) * triton.cdiv(n, cfg[0]) <= SK_WS:
            return _streamk(x.contiguous(), w_full, n, state, cfg)
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
