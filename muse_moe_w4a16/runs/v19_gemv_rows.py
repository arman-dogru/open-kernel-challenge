# okc: problem=muse_moe_w4a16
"""v19 (= v18 with row-heavy GEMV tiles: one word per lane along k, several rows per thread,
2-D tiles with one accumulator per route (v18's 3-D [route, row, word] tiles split warps over routes and
converted layouts through shared memory; and it issued ~1 x load per weight).
v18: small-batch routed experts as a grouped split-K GEMV on CUDA cores with packed dequant;
Marlin (the board's v9b path, unchanged) above SMALL_MAX tokens.

GEMV path (bs <= 32):
- prepare() keeps the 32 local experts' E2M1 codes as int64 words of 16 codes with a fixed,
  lossless bit permutation inside each 32-bit half (as v11-v17); E4M3 scales and FP32 global
  scales unchanged. NVFP4 stays NVFP4.
- Dequant: per 32-bit half, 4 x (funnel-shift rotate + AND [+ lop3]) give 4 bf16x2 holding
  e2m1 * 2^-126; one mul.rn.bf16x2 by scale * 2^119 gives e2m1 * scale * 2^-7, exact (<= 6
  significant bits, normal range). Products and sums in FP32 on CUDA cores (no tensor cores:
  1-8 routes per expert), global scale * 2^7 in the epilogue.
- One routing-prep kernel builds per-expert route lists and a compact work list of chunks of
  MB routes of one expert (fixed-size buffers, graph-safe); grid = (row blocks, work bound,
  split-K) and programs past the device-side work count exit at once.
- GEMM 1 fuses gate and up rows; split-K partials go to a zeroed FP32 buffer by atomics, then
  silu(gate) * up is stored as BF16 (as SGLang). GEMM 2 accumulates FP32 per route; an FP32
  post-norm + weighted sum kernel combines the local routes of each token."""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl

try:
    from sgl_kernel.scalar_type import scalar_types

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        _post_expert_norm_weighted_sum_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
except ImportError:  # CPU interpreter tests of the GEMV path only
    pass

POST_NORM_EPS = 1e-8

# Where code j's bits (bit0 = mantissa, bit1/bit2 = exponent, bit3 = sign) go in the permuted
# 32-bit word. Pairs (0,1), (2,3), (4,5) come out of rotl(w, 6 / 2 / 28) & 0x81C081C0; pair (6,7)
# is (w & 0x81808180) | (rotl(w, 24) & 0x00400040).
_BITPOS = (
    (0, 1, 2, 9), (16, 17, 18, 25),      # rotl 6
    (4, 5, 6, 13), (20, 21, 22, 29),     # rotl 2
    (10, 11, 12, 19), (26, 27, 28, 3),   # rotl 28
    (14, 7, 8, 15), (30, 23, 24, 31),    # in place + mantissa from rotl 24
)

# ins: $16 = int64 word (16 codes: low half k0..7, high half k8..15), $17 = bf16 scale * 2^119.
# outs: $0..$15 = bf16 k0..k15.
_DEQ_ASM = tl.constexpr("""
{
.reg .b32 lo, hi, s2, r, t;
mov.b64 {lo, hi}, $16;
mov.b32 s2, {$17, $17};
shf.l.wrap.b32 r, lo, lo, 6;
and.b32 r, r, 0x81C081C0;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$0, $1}, r;
shf.l.wrap.b32 r, lo, lo, 2;
and.b32 r, r, 0x81C081C0;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$2, $3}, r;
shf.l.wrap.b32 r, lo, lo, 28;
and.b32 r, r, 0x81C081C0;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$4, $5}, r;
shf.l.wrap.b32 t, lo, lo, 24;
and.b32 t, t, 0x00400040;
lop3.b32 r, lo, 0x81808180, t, 0xEA;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$6, $7}, r;
shf.l.wrap.b32 r, hi, hi, 6;
and.b32 r, r, 0x81C081C0;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$8, $9}, r;
shf.l.wrap.b32 r, hi, hi, 2;
and.b32 r, r, 0x81C081C0;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$10, $11}, r;
shf.l.wrap.b32 r, hi, hi, 28;
and.b32 r, r, 0x81C081C0;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$12, $13}, r;
shf.l.wrap.b32 t, hi, hi, 24;
and.b32 t, t, 0x00400040;
lop3.b32 r, hi, 0x81808180, t, 0xEA;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$14, $15}, r;
}
""")


@triton.jit
def _rotl(w, n: tl.constexpr):
    return (w << n) | (w >> (32 - n))


@triton.jit
def _pair_native(r, s):
    """CPU-interpreter twin of the asm for one bf16x2: (low, high) as fp32 * scale."""
    lo = ((r & 0xFFFF) << 16).to(tl.float32, bitcast=True)
    hi = (r & 0xFFFF0000).to(tl.float32, bitcast=True)
    return lo * s, hi * s


@triton.jit
def _post_norm_sum_kernel(y_ptr, nw_ptr, tw_ptr, ids_ptr, out_ptr, eps,
                          TOPK: tl.constexpr, H: tl.constexpr):
    """out[t] = sum_k w[t,k] * rmsnorm(y[t,k] * (1 + norm_w)) over local slots, FP32 math."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    nw = tl.load(nw_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        sid = tl.load(ids_ptr + t * TOPK + k)
        if sid >= 0:
            y = tl.load(y_ptr + (t * TOPK + k).to(tl.int64) * H + offs) * nw
            r = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * (r * tl.load(tw_ptr + t * TOPK + k))
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty))


def permute_codes(w):
    """uint8 [..., K/2] packed E2M1 (low nibble = even k) -> int64 [..., K/16], each 32-bit half
    holding 8 codes with their bits moved to _BITPOS (a lossless bit permutation)."""
    w32 = w.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w32)
    for j, pos in enumerate(_BITPOS):
        for b, p in enumerate(pos):
            out |= ((w32 >> (4 * j + b)) & 1) << p
    out = torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)
    return out.view(torch.int64)


# ---------------------------------------------------------------------------------------------
# Small-batch path (bs <= SMALL_MAX): grouped split-K GEMV on CUDA cores, packed dequant.
# ---------------------------------------------------------------------------------------------
@triton.jit
def _route_prep(ids_ptr, slot_ptr, cnt_ptr, list_ptr, work_ptr, nwork_ptr, NR, local_start,
                COUNT: tl.constexpr, MAXM: tl.constexpr, MB: tl.constexpr, BR: tl.constexpr):
    """One program. Per local expert: its route list (route = token * top_k + slot) and count;
    a compact work list of (expert, first list index) chunks of MB routes; slot ids for the
    post-norm sum (local expert or -1)."""
    r = tl.arange(0, BR)
    inb = r < NR
    e = tl.load(ids_ptr + r, mask=inb, other=-1) - local_start
    loc = inb & (e >= 0) & (e < COUNT)
    tl.store(slot_ptr + r, tl.where(loc, e, -1), mask=inb)
    CH: tl.constexpr = MAXM // MB
    ch = tl.arange(0, CH)
    base = 0
    for ex in range(COUNT):
        sel = loc & (e == ex)
        n = tl.sum(sel.to(tl.int32), axis=0)
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(list_ptr + ex * MAXM + pos, r, mask=sel)
        tl.store(cnt_ptr + ex, n)
        nch = (n + MB - 1) // MB
        tl.store(work_ptr + base + ch, ex * MAXM + ch * MB, mask=ch < nch)
        base += nch
    tl.store(nwork_ptr, base)


@triton.jit
def _gemv(x_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, cnt_ptr, list_ptr, work_ptr, nwork_ptr, local_start, MAXM,
          N: tl.constexpr, K: tl.constexpr, XROW_DIV: tl.constexpr, GATED: tl.constexpr,
          ATOMIC: tl.constexpr, MB: tl.constexpr, SPLIT: tl.constexpr,
          BN: tl.constexpr, BKW: tl.constexpr, NATIVE: tl.constexpr):
    """out[route, n] = sum_k x[route // XROW_DIV, k] * W_e[n, k] for up to MB (1, 2 or 4) routes
    of one expert. W_e: int64 words [ROWS, K/16] of permuted E2M1 codes (16 k per word; gate rows
    [0, N), up rows [N, 2N) when GATED), E4M3 scales [ROWS, K/16] (one per word). Each word is
    dequantized once and used for all MB routes (2-D tiles, one accumulator per route).
    FP32 products and sums. ATOMIC: split-K partials added (FP32) into a zeroed [routes, ROWS]
    buffer; else a direct store (silu(gate) * up in out's dtype when GATED)."""
    pid_n = tl.program_id(0)
    wk = tl.program_id(1)
    pid_s = tl.program_id(2)
    if wk >= tl.load(nwork_ptr):
        return
    item = tl.load(work_ptr + wk)
    ex = item // MAXM
    i0 = item % MAXM
    n_e = tl.load(cnt_ptr + ex)
    lbase = list_ptr + ex * MAXM + i0
    r0 = tl.load(lbase)
    v0 = i0 < n_e
    v1 = (i0 + 1) < n_e
    v2 = (i0 + 2) < n_e
    v3 = (i0 + 3) < n_e
    r1 = tl.load(lbase + 1, mask=v1, other=0)
    r2 = tl.load(lbase + 2, mask=v2, other=0)
    r3 = tl.load(lbase + 3, mask=v3, other=0)
    KW: tl.constexpr = K // 16
    KWS: tl.constexpr = KW // SPLIT
    if GATED:
        ROWS: tl.constexpr = 2 * N
    else:
        ROWS: tl.constexpr = N
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_w = tl.arange(0, BKW)
    kw0 = pid_s * KWS
    w_off = ex.to(tl.int64) * ROWS * KW + offs_n[:, None] * KW + kw0 + offs_w[None, :]   # [BN, BKW]
    xk = (kw0 + offs_w[None, :]) * 16                                                     # [1, BKW]
    x0 = x_ptr + (r0 // XROW_DIV).to(tl.int64) * K + xk
    x1 = x_ptr + (r1 // XROW_DIV).to(tl.int64) * K + xk
    x2 = x_ptr + (r2 // XROW_DIV).to(tl.int64) * K + xk
    x3 = x_ptr + (r3 // XROW_DIV).to(tl.int64) * K + xk
    g0 = tl.zeros((BN, BKW), dtype=tl.float32)
    g1 = tl.zeros((BN, BKW), dtype=tl.float32)
    g2 = tl.zeros((BN, BKW), dtype=tl.float32)
    g3 = tl.zeros((BN, BKW), dtype=tl.float32)
    u0 = tl.zeros((BN, BKW), dtype=tl.float32)
    u1 = tl.zeros((BN, BKW), dtype=tl.float32)
    u2 = tl.zeros((BN, BKW), dtype=tl.float32)
    u3 = tl.zeros((BN, BKW), dtype=tl.float32)
    for kk in range(0, KWS, BKW):
        q = tl.load(w_ptr + w_off + kk)
        sc = tl.load(sf_ptr + w_off + kk).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        o = _deq(q, sc, NATIVE)
        g0 = _fma16(g0, o, x0 + kk * 16)
        if MB > 1:
            g1 = _fma16(g1, o, x1 + kk * 16)
        if MB > 2:
            g2 = _fma16(g2, o, x2 + kk * 16)
            g3 = _fma16(g3, o, x3 + kk * 16)
        if GATED:
            q = tl.load(w_ptr + w_off + N * KW + kk)
            sc = tl.load(sf_ptr + w_off + N * KW + kk).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            o = _deq(q, sc, NATIVE)
            u0 = _fma16(u0, o, x0 + kk * 16)
            if MB > 1:
                u1 = _fma16(u1, o, x1 + kk * 16)
            if MB > 2:
                u2 = _fma16(u2, o, x2 + kk * 16)
                u3 = _fma16(u3, o, x3 + kk * 16)
    gs = tl.load(gs_ptr + local_start + ex) * 128.0  # undoes the dequant's 2^-7
    nm = offs_n < N
    _out(out_ptr, r0, g0, u0, gs, offs_n, nm, N, ROWS, GATED, ATOMIC)
    if MB > 1:
        _out(out_ptr, r1, g1, u1, gs, offs_n, nm & ((offs_n * 0 + v1.to(tl.int32)) > 0), N, ROWS, GATED, ATOMIC)
    if MB > 2:
        _out(out_ptr, r2, g2, u2, gs, offs_n, nm & ((offs_n * 0 + v2.to(tl.int32)) > 0), N, ROWS, GATED, ATOMIC)
        _out(out_ptr, r3, g3, u3, gs, offs_n, nm & ((offs_n * 0 + v3.to(tl.int32)) > 0), N, ROWS, GATED, ATOMIC)


@triton.jit
def _out(out_ptr, r, accg, accu, gs, offs_n, m, N: tl.constexpr, ROWS: tl.constexpr,
         GATED: tl.constexpr, ATOMIC: tl.constexpr):
    g = tl.sum(accg, axis=1) * gs
    if ATOMIC:
        p = out_ptr + r.to(tl.int64) * ROWS + offs_n
        tl.atomic_add(p, g, mask=m, sem="relaxed")
        if GATED:
            tl.atomic_add(p + N, tl.sum(accu, axis=1) * gs, mask=m, sem="relaxed")
    else:
        p = out_ptr + r.to(tl.int64) * N + offs_n
        if GATED:
            res = g * tl.sigmoid(g) * (tl.sum(accu, axis=1) * gs)
        else:
            res = g
        tl.store(p, res.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _deq(q, sc, NATIVE: tl.constexpr):
    """int64 words of 16 permuted codes + their E4M3 scale -> 16 bf16 (k order) = e2m1 * scale * 2^-7."""
    s = sc * 6.646139978924579e+35  # 2^119: e2m1 (as bf16 * 2^-126) * scale * 2^119 = value * 2^-7
    if NATIVE:
        lo = (q & 0xFFFFFFFF).to(tl.uint32)
        hi = ((q >> 32) & 0xFFFFFFFF).to(tl.uint32)
        o0, o1 = _pair_native(_rotl(lo, 6) & 0x81C081C0, s)
        o2, o3 = _pair_native(_rotl(lo, 2) & 0x81C081C0, s)
        o4, o5 = _pair_native(_rotl(lo, 28) & 0x81C081C0, s)
        o6, o7 = _pair_native((lo & 0x81808180) | (_rotl(lo, 24) & 0x00400040), s)
        o8, o9 = _pair_native(_rotl(hi, 6) & 0x81C081C0, s)
        o10, o11 = _pair_native(_rotl(hi, 2) & 0x81C081C0, s)
        o12, o13 = _pair_native(_rotl(hi, 28) & 0x81C081C0, s)
        o14, o15 = _pair_native((hi & 0x81808180) | (_rotl(hi, 24) & 0x00400040), s)
        return o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    else:
        return tl.inline_asm_elementwise(
            _DEQ_ASM, "=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,l,h", [q, s.to(tl.bfloat16)],
            dtype=(tl.bfloat16,) * 16, is_pure=True, pack=1)


@triton.jit
def _fma16(acc, o, xp):
    """acc[n, w] += sum_j o[j][n, w] * x[16 w + j] (an unused route slot reads route 0's row; its
    result is masked at the store)."""
    for j in tl.static_range(16):
        acc += o[j].to(tl.float32) * tl.load(xp + j).to(tl.float32)
    return acc


@triton.jit
def _silu_mul_kernel(h_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """out[r, :] = silu(h[r, :N]) * h[r, N:] (FP32 math, stored in out's dtype)."""
    r = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(h_ptr + r * (2 * N) + offs)
    u = tl.load(h_ptr + r * (2 * N) + N + offs)
    tl.store(out_ptr + r * N + offs, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty))


SMALL_MAX = 32  # token counts up to this use the GEMV path; larger ones use Marlin (v9b)
# (MB, w13 (BN, BKW, SPLIT, warps), w2 (BN, BKW, SPLIT, warps)) per scored token count.
SMALL_CFG = {1: (1, (32, 64, 4, 4), (32, 64, 4, 4)), 8: (2, (32, 64, 4, 4), (32, 64, 4, 4)),
             16: (4, (32, 64, 4, 4), (32, 64, 4, 4)), 32: (4, (32, 64, 4, 4), (32, 64, 4, 4))}
LOCAL = (0, 32)


def _small_cfg(bs):
    return SMALL_CFG[min(SMALL_CFG, key=lambda b: (abs(b - bs), b))]


def _kernel_small(data, state, cfg_override=None):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    bs, H = x.shape
    I = cfg["d_expert"]
    top_k = cfg["top_k"]
    start, count = cfg["local_start"], cfg["local_count"]
    MB, c1, c2 = cfg_override or _small_cfg(bs)
    NR = bs * top_k
    MAXM = triton.next_power_of_2(NR)
    MB = min(MB, MAXM)
    assert MB in (1, 2, 4)
    nwork = min(NR, NR // MB + min(count, NR))
    dev = x.device
    ib = torch.empty((NR + count + count * MAXM + nwork + 1,), device=dev, dtype=torch.int32)
    slot = ib[:NR]
    cnt = ib[NR:NR + count]
    lst = ib[NR + count:NR + count + count * MAXM]
    work = ib[NR + count + count * MAXM:NR + count + count * MAXM + nwork]
    nw = ib[NR + count + count * MAXM + nwork:]
    _route_prep[(1,)](data["topk_ids"].contiguous(), slot, cnt, lst, work, nw, NR, start,
                      COUNT=count, MAXM=MAXM, MB=MB, BR=triton.next_power_of_2(NR), num_warps=4)
    native = state["native"]
    BN1, BKW1, S1, W1 = c1
    BN2, BKW2, S2, W2 = c2
    inter = torch.empty((NR, I), device=dev, dtype=x.dtype)
    if S1 > 1:
        h32 = torch.zeros((NR, 2 * I), device=dev, dtype=torch.float32)
        k1 = _gemv[(I // BN1, nwork, S1)](x, state["w13"], state["w13_sf"], state["w13_gs"], h32, cnt, lst, work, nw,
                                    start, MAXM, N=I, K=H, XROW_DIV=top_k, GATED=True, ATOMIC=True, MB=MB,
                                    SPLIT=S1, BN=BN1, BKW=BKW1, NATIVE=native, num_warps=W1)
        sb = min(1024, I)
        _silu_mul_kernel[(NR, I // sb)](h32, inter, N=I, BLOCK=sb, num_warps=4)
    else:
        k1 = _gemv[(I // BN1, nwork, 1)](x, state["w13"], state["w13_sf"], state["w13_gs"], inter, cnt, lst, work, nw,
                                   start, MAXM, N=I, K=H, XROW_DIV=top_k, GATED=True, ATOMIC=False, MB=MB,
                                   SPLIT=1, BN=BN1, BKW=BKW1, NATIVE=native, num_warps=W1)
    if S2 > 1:
        y32 = torch.zeros((NR, H), device=dev, dtype=torch.float32)
    else:
        y32 = torch.empty((NR, H), device=dev, dtype=torch.float32)
    k2 = _gemv[(H // BN2, nwork, S2)](inter, state["w2"], state["w2_sf"], state["w2_gs"], y32, cnt, lst, work, nw,
                                start, MAXM, N=H, K=I, XROW_DIV=1, GATED=False, ATOMIC=S2 > 1, MB=MB,
                                SPLIT=S2, BN=BN2, BKW=BKW2, NATIVE=native, num_warps=W2)
    if BENCH:
        LAST[:] = [k1, k2]
    out = torch.empty((bs, H), device=dev, dtype=x.dtype)
    _post_norm_sum_kernel[(bs,)](y32, state["norm_w"], data["topk_weights"].float().contiguous(), slot, out,
                                 POST_NORM_EPS, TOPK=top_k, H=H, num_warps=8)
    return out


def _permute_local(w, start, count, chunk=4):
    """Permuted int64 code words for experts [start, start + count) only."""
    out = torch.empty((count, w.shape[1], w.shape[2] // 8), dtype=torch.int64, device=w.device)
    for e in range(0, count, chunk):
        out[e:e + chunk] = permute_codes(w[start + e:start + min(e + chunk, count)])
    return out


def prepare(weights):
    start, count = LOCAL
    sl = slice(start, start + count)
    native = weights["w13"].device.type == "cpu"
    st = {} if native else _prepare_marlin(weights)
    st.update({
        "w13": _permute_local(weights["w13"], start, count),
        "w13_sf": weights["w13_sf"][sl].contiguous().view(torch.uint8),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": _permute_local(weights["w2"], start, count),
        "w2_sf": weights["w2_sf"][sl].contiguous().view(torch.uint8),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "native": native,
    })
    if not native:
        torch.cuda.synchronize()
        if BENCH:
            _bench(st, weights)
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    assert (cfg["local_start"], cfg["local_count"]) == LOCAL
    if state["native"] or data["hidden_states"].shape[0] <= SMALL_MAX:
        return _kernel_small(data, state)
    return _kernel_marlin(data, state)


# EXPLORATION ONLY (check runs): BENCH times GEMV configs against the Marlin path inside
# prepare() on synthetic routings and prints the results; nothing it measures is kept or reused
# by kernel_fn (which only reads the fixed SMALL_CFG table). A submit sets BENCH = False.
BENCH = True
_BENCHED = []
BENCH_TILES = [(32, 64, 4, 4), (64, 64, 4, 4), (32, 64, 2, 4), (64, 64, 2, 8), (16, 64, 4, 2),
               (32, 32, 4, 4), (64, 32, 8, 4)]
BENCH_MB = {1: (1,), 8: (2, 4), 16: (2, 4), 32: (4,)}
LAST = []


def _synthetic(bs, gen, dev, I, H):
    """Heavy-tailed expert popularity, top-8 of 256, busiest group of 32 rotated to 0-31."""
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.5 * z / (-u.log()).sqrt()
    r = scores.topk(8, dim=-1)
    ids = r.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    x = torch.randn((bs, H), generator=gen, device=dev).to(torch.bfloat16)
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (r.values / r.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": H, "d_expert": I, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def _time(fn, flush, reps=30):
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


def _bench(st, weights):
    if _BENCHED:
        return
    _BENCHED.append(1)
    import time
    t0 = time.time()
    dev = weights["w13"].device
    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    flush = torch.empty((512 * 1024 * 1024 // 4,), device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(1234)
    for bs in (1, 8, 16, 32):
        data = _synthetic(bs, gen, dev, I, H)
        loc = (data["topk_ids"] < 32)
        nexp = int(torch.unique(data["topk_ids"][loc]).numel())
        tm, ref = _time(lambda: _kernel_marlin(data, st), flush)
        print(f"BENCH bs={bs} local_routes={int(loc.sum())} experts={nexp} marlin={tm:.1f}us", flush=True)
        rows = []
        for MB in BENCH_MB[bs]:
            for tile in BENCH_TILES:
                if time.time() - t0 > 300:
                    print("BENCH time cap", flush=True)
                    return
                try:
                    t, out = _time(lambda: _kernel_small(data, st, (MB, tile, tile)), flush)
                    err = float((out.float() - ref.float()).norm() / ref.float().norm())
                    try:
                        rr = f"regs={LAST[0].n_regs}/{LAST[1].n_regs} spills={LAST[0].n_spills}/{LAST[1].n_spills}"
                    except Exception:
                        rr = ""
                    rows.append((t, MB, tile, err, rr))
                except Exception as ex:  # report and continue
                    print(f"BENCH bs={bs} MB={MB} tile={tile} error {type(ex).__name__}: {str(ex)[:120]}", flush=True)
        rows.sort()
        for t, MB, tile, err, rr in rows:
            print(f"BENCH bs={bs} MB={MB} tile={tile} {t:.1f}us err_vs_marlin={err:.4f} {rr}", flush=True)
        # mixed: best w13 tile with each w2 tile, at the best MB
        if rows:
            _, MB, best, _, _ = rows[0]
            for tile in BENCH_TILES:
                t, _ = _time(lambda: _kernel_small(data, st, (MB, best, tile)), flush)
                print(f"BENCH bs={bs} MB={MB} w13={best} w2={tile} {t:.1f}us", flush=True)
    print(f"BENCH done in {time.time() - t0:.0f}s", flush=True)


# --- Marlin path (v9b_resubmit.py, unchanged apart from the two function names) ---
class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8
# Token count -> Marlin moe_block_size, measured best (SGLang's heuristic picks 48 and 64 here).
BLOCK_M_OVERRIDE = {128: 16, 256: 32}


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _prepare_marlin(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def _kernel_marlin(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    block_size_m = BLOCK_M_OVERRIDE.get(num_tokens, block_size_m)
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _post_norm_sum(y, topk_weights, state["norm_w"], local_ids)


def _post_norm_sum(y, topk_weights, norm_w, slot_ids):
    """SGLang's fused post-expert norm + weighted sum kernel, launched directly at any token count
    (fused_post_expert_norm_weighted_sum's own launch, minus its < 32-token PyTorch fallback)."""
    tokens, topk, hidden = y.shape
    x = y.contiguous()
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x, norm_w, topk_weights.contiguous(), out, slot_ids,
        TOPK=topk, H=hidden, eps=POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True,
        HAS_IDS=True, BLOCK_H=block_h, num_warps=8 if block_h >= 4096 else 4, num_stages=2)
    return out
