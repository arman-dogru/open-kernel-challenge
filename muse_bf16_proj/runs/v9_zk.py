# okc: problem=muse_bf16_proj
"""v9: lossless 12-bit weight compression for the BF16 layers at m <= 32 (else as v8).

The small-m BF16 layers only stream weights, so their time is bytes / bandwidth. prepare() stores
each BF16 weight losslessly in 12 bits (the rules allow lossless compression):
- L [N, K] uint8: the low byte of the BF16 bit pattern (exponent bit 0 + 7 mantissa bits), as is.
- C [N, K/2] uint8: a 4-bit code per weight, sign << 3 | (exponent >> 1) - B, for the 8 values of
  exponent >> 1 (16 binades) starting at the per-layer base B that covers the most weights. Two
  codes per byte: within each 256-wide K chunk, byte j holds k = j (low nibble) and k = j + 128.
- Exceptions: weights outside those 16 binades (randn / sqrt(K) weights: ~1e-4 of them; zeros
  are exceptions too) are kept exactly, as (row, k, value) lists sorted by (32-row block, 64-k
  chunk) with CSR offsets. If more than 0.1% of a layer's weights would be exceptions, that layer
  is not compressed (prepare() decides from the weights alone) and runs v8's path.
Decoding (inline PTX, 12 int ops per 8 weights, straight into the MMA's A operand) rebuilds the
BF16 pattern with exponent >> 1 = B0 + idx (a fixed B0), i.e. the weight times 2^(2 (B0 - B))
exactly. Exception slots decode to +-2^(2 B0 - 127); after the K loop each exception adds
w_exact * x - w_decoded * x (exact BF16 x BF16 products, FP32 adds). The FP32 tile is then scaled
by 2^(2 (B - B0)) (exact) before the BF16 store / FP32 atomics. So: BF16 weights and activations,
exact products, FP32 accumulation (in a different order), one BF16 rounding at the end.
Same stream-K / self-cleaning scratch as v8; one program per row block (full K, no atomics) when
there are at least 3/4 as many row blocks as SMs.

v8: stream-K for the big small-m layers (qkv_gate, pooler, unpooler at m <= 32), else as v7.

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


ZK_B0 = 60          # decoded exponent >> 1 = ZK_B0 + idx: decoded values in [2^-7, 2^9)
ZK_BK = 256         # K chunk of the code pairing (fixed at prepare time)
ZK_CH = 64          # exception CSR granularity along K
ZK_RB = 32          # exception CSR granularity along N
ZK_MAX_EXC = 1e-3   # compress a layer only if at most this fraction of weights are exceptions

# $4 = 4 code bytes (low nibble: k = j .. j+3 of the chunk's first half, high nibble: k + 128),
# $5 / $6 = the 4 low bytes of the first / second half. Out: $0,$1 = 4 BF16 of the first half,
# $2,$3 = 4 BF16 of the second half. Byte i of every input belongs to output element i.
_ZK_ASM = tl.constexpr("""
{
.reg .b32 ta, tb, ha, hb, sa;
and.b32 ta, $4, 0x07070707;
add.u32 ta, ta, 0x3C3C3C3C;
shl.b32 sa, $4, 4;
lop3.b32 ha, sa, 0x80808080, ta, 0xEA;
shr.u32 tb, $4, 4;
and.b32 tb, tb, 0x07070707;
add.u32 tb, tb, 0x3C3C3C3C;
lop3.b32 hb, $4, 0x80808080, tb, 0xEA;
prmt.b32 $0, $5, ha, 0x5140;
prmt.b32 $1, $5, ha, 0x7362;
prmt.b32 $2, $6, hb, 0x5140;
prmt.b32 $3, $6, hb, 0x7362;
}
""")


@triton.jit
def _zk_native(code, low):
    """CPU-interpreter twin of one half: code (4-bit) + low byte -> the decoded value (fp32)."""
    c = code.to(tl.int32)
    hi = ((c & 8) << 4) | ((c & 7) + 60)
    bits = (hi << 8) | low.to(tl.int32)
    return (bits << 16).to(tl.float32, bitcast=True)


@triton.jit
def _zk_dec(c, la, lb, NATIVE: tl.constexpr):
    if NATIVE:
        return _zk_native(c & 15, la), _zk_native(c >> 4, lb)
    else:
        return tl.inline_asm_elementwise(_ZK_ASM, "=r,=r,=r,=r,r,r,r", [c, la, lb],
                                         dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)


@triton.jit
def _zk_gemv(x_ptr, l_ptr, c_ptr, out_ptr, ws_ptr, cnt_ptr, eoff_ptr, erk_ptr, evt_ptr, evd_ptr,
             M, N, K, stride_xm, stride_om, scale, TOTAL, NPROG,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, CH: tl.constexpr,
             RB: tl.constexpr, NATIVE: tl.constexpr):
    """Stream-K small-m GEMM over the 12-bit compressed weights (see the module docstring)."""
    pid = tl.program_id(0)
    upr = K // BK
    nch = K // CH
    start = (pid * TOTAL) // NPROG
    end = ((pid + 1) * TOTAL) // NPROG
    offs_m = tl.arange(0, BM)
    offs_b = tl.arange(0, BN)
    offs_j = tl.arange(0, BK // 2)
    m_mask = offs_m < M
    u = start
    while u < end:
        rb = u // upr
        kb0 = u - rb * upr
        kb1 = tl.minimum(upr, kb0 + (end - u))
        offs_n = rb * BN + offs_b
        n_mask = offs_n < N
        l_ptrs = l_ptr + offs_n[:, None] * K + offs_j[None, :]
        c_ptrs = c_ptr + offs_n[:, None] * (K // 2) + offs_j[None, :]
        x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_j[:, None]
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        for kb in range(kb0, kb1):
            c = tl.load(c_ptrs + kb * (BK // 2), mask=n_mask[:, None], other=0)
            la = tl.load(l_ptrs + kb * BK, mask=n_mask[:, None], other=0)
            lb = tl.load(l_ptrs + kb * BK + BK // 2, mask=n_mask[:, None], other=0)
            xa = tl.load(x_ptrs + kb * BK, mask=m_mask[None, :], other=0.0)
            xb = tl.load(x_ptrs + kb * BK + BK // 2, mask=m_mask[None, :], other=0.0)
            wa, wb = _zk_dec(c, la, lb, NATIVE)
            acc = tl.dot(wa.to(xa.dtype), xa, acc)
            acc = tl.dot(wb.to(xb.dtype), xb, acc)
        # exceptions of this segment: rows of each RB-row block, K chunks [kb0, kb1) * BK / CH
        for s in tl.static_range(BN // RB):
            key = (rb * (BN // RB) + s) * nch
            e0 = tl.load(eoff_ptr + key + kb0 * (BK // CH))
            e1 = tl.load(eoff_ptr + key + kb1 * (BK // CH))
            for e in range(e0, e1):
                rk = tl.load(erk_ptr + e)
                r = s * RB + (rk >> 16)
                kk = rk & 0xFFFF
                xr = tl.load(x_ptr + offs_m * stride_xm + kk, mask=m_mask, other=0.0).to(tl.float32)
                vt = tl.load(evt_ptr + e)
                vd = tl.load(evd_ptr + e)
                acc += tl.where(offs_b[:, None] == r, (vt * xr - vd * xr)[None, :], 0.0)
        acc = acc * scale
        o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
        o_mask = m_mask[None, :] & n_mask[:, None]
        nu = kb1 - kb0
        if nu == upr:
            tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)
        else:
            ws_ptrs = ws_ptr + rb * (BN * BM) + offs_b[:, None] * BM + offs_m[None, :]
            tl.atomic_add(ws_ptrs, acc, sem="relaxed")
            tl.debug_barrier()
            old = tl.atomic_add(cnt_ptr + rb, nu, sem="acq_rel")
            if old + nu == upr:
                tot = tl.atomic_add(ws_ptrs, tl.zeros((BN, BM), dtype=tl.float32), sem="acquire")
                tl.store(o_ptrs, tot.to(out_ptr.dtype.element_ty), mask=o_mask)
                tl.store(ws_ptrs, tl.zeros((BN, BM), dtype=tl.float32))
                tl.store(cnt_ptr + rb, 0)
        u += nu


def _zk_compress(w):
    """BF16 [N, K] -> the 12-bit form (module docstring), or None if it doesn't fit."""
    n, k = w.shape
    if n % ZK_RB or k % ZK_BK or k >= 65536:
        return None
    b = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    ehi = (b >> 8) & 0x7F
    hist = torch.bincount(ehi.flatten(), minlength=128).cpu()
    win = torch.stack([hist[i:i + 8].sum() for i in range(121)])
    base = int(win.argmax())
    n_exc = n * k - int(win[base])
    if n_exc > ZK_MAX_EXC * n * k:
        return None
    idx = ehi - base
    inr = (idx >= 0) & (idx <= 7)
    sign = (b >> 15) & 1
    code = torch.where(inr, (sign << 3) | idx, sign << 3)
    low = torch.where(inr, b & 0xFF, torch.zeros_like(b))
    cc = code.view(n, k // ZK_BK, 2, ZK_BK // 2)
    cbytes = (cc[:, :, 0] | (cc[:, :, 1] << 4)).reshape(n, k // 2).to(torch.uint8).contiguous()
    lbytes = low.to(torch.uint8).contiguous()
    # exceptions, sorted by (RB-row block, CH chunk, row, k)
    rows, cols = torch.nonzero(~inr, as_tuple=True)
    key = (rows // ZK_RB) * (k // ZK_CH) + cols // ZK_CH
    order = torch.argsort(key * (ZK_RB * ZK_CH) + (rows % ZK_RB) * ZK_CH + cols % ZK_CH)
    rows, cols, key = rows[order], cols[order], key[order]
    nkeys = (n // ZK_RB) * (k // ZK_CH)
    eoff = torch.zeros(nkeys + 1, dtype=torch.int64, device=w.device)
    eoff[1:] = torch.cumsum(torch.bincount(key, minlength=nkeys), 0)
    shift = 2.0 ** (2 * (ZK_B0 - base))
    vt = w[rows, cols].float() * shift
    vd = torch.where(sign[rows, cols] == 1, -1.0, 1.0) * 2.0 ** (2 * ZK_B0 - 127)
    return dict(l=lbytes, c=cbytes, eoff=eoff.to(torch.int32).contiguous(),
                erk=(((rows % ZK_RB) << 16) | cols).to(torch.int32).contiguous(),
                evt=vt.float().contiguous(), evd=vd.float().contiguous(),
                scale=2.0 ** (2 * (base - ZK_B0)), n_exc=n_exc, base=base)


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


# Compressed layers at m <= 32: (BN, programs per SM (stream-K), num_warps, num_stages).
# Check-only sweep: one config per token count while exploring.
ZK_SWEEP = {1: (32, 2, 2, 3), 8: (64, 1, 4, 3), 16: (32, 2, 2, 4), 32: (64, 1, 4, 4)}
ZK_CONFIG = (32, 2, 2, 3)


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
    state["zk"] = {name: _zk_compress(w) for name, w in weights.items() if w.dtype == torch.bfloat16}
    return state


def _zk(x, z, n, state, cfg):
    bn, per_sm, warps, stages = cfg
    m, k = x.shape
    bm = _bm(m)
    row_blocks = triton.cdiv(n, bn)
    upr = k // ZK_BK
    total = row_blocks * upr
    n_sm = state["n_sm"]
    nprog = row_blocks if 4 * row_blocks >= 3 * n_sm else min(total, n_sm * per_sm)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _zk_gemv[(nprog,)](x, z["l"], z["c"], out, state["sk_ws"], state["sk_cnt"], z["eoff"], z["erk"],
                       z["evt"], z["evd"], m, n, k, x.stride(0), out.stride(0), z["scale"], total, nprog,
                       BM=bm, BN=bn, BK=ZK_BK, CH=ZK_CH, RB=ZK_RB, NATIVE=(x.device.type == "cpu"),
                       num_warps=warps, num_stages=stages)
    return out


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
    z = state["zk"].get(layer)
    if z is not None and m <= 32:
        cfg = ZK_SWEEP.get(m, ZK_CONFIG)
        if cfg[0] * _bm(m) * triton.cdiv(n, cfg[0]) <= SK_WS and n % ZK_RB == 0:
            return _zk(x.contiguous(), z, n, state, cfg)
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
