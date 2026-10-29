# okc: problem=muse_moe_w4a16
"""Routed experts (NVFP4 W4A16) as two Triton grouped GEMMs (swap-AB: weight rows are the MMA's
M side, the <=16 routes of a sorted block its N side), with a packed register dequant.

Dequant (the part that made the earlier Triton version ALU-bound at ~13 ops per weight):
- prepare() permutes the bits inside each 32-bit word of E2M1 codes (8 codes, lossless, a fixed
  bit permutation per word; the E4M3 block scales and FP32 global scales are untouched).
  The layout is chosen so that each pair of codes becomes a bf16x2 by one funnel-shift rotate
  and one AND (3 ops for the fourth pair): the code's magnitude bits land on bf16 bits 8..6
  and its sign on bit 15, i.e. the bf16 value is e2m1 * 2^-126 (exact; 0.5 is a subnormal).
- One mul.rn.bf16x2 by the block scale pre-multiplied by 2^119 (exact in bf16) gives
  e2m1 * scale * 2^-7, exact (at most 6 significant bits, in the normal range).
  That is ~1.6 integer/FP ops per weight instead of ~13.
- BF16 operands into tensor-core dots with FP32 accumulation; the epilogue multiplies by
  global scale * 2^7 in FP32. GEMM 1 fuses gate and up and silu(gate) * up (BF16 intermediate,
  as SGLang); GEMM 2 writes BF16 per route; SGLang's Triton post-expert norm + weighted sum
  kernel combines the routes (launched directly, as in v9).
- Routing: SGLang's moe_align_block_size over the 32 local experts (+1 skipped bucket for the
  routes that go to other GPUs), graph-safe; empty blocks exit immediately."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2]
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H]


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


POST_NORM_EPS = 1e-8
BM = 16
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
def _interleave_all(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15):
    """16 same-shape tensors -> [..., 16] with t_i at index i (in-register joins)."""
    a0 = tl.join(t0, t8)
    a1 = tl.join(t4, t12)
    a2 = tl.join(t2, t10)
    a3 = tl.join(t6, t14)
    a4 = tl.join(t1, t9)
    a5 = tl.join(t5, t13)
    a6 = tl.join(t3, t11)
    a7 = tl.join(t7, t15)
    b0 = tl.interleave(a0, a1)
    b1 = tl.interleave(a2, a3)
    b2 = tl.interleave(a4, a5)
    b3 = tl.interleave(a6, a7)
    c0 = tl.interleave(b0, b1)
    c1 = tl.interleave(b2, b3)
    return tl.interleave(c0, c1)


@triton.jit
def _deq16(q, sc, DT: tl.constexpr, NATIVE: tl.constexpr):
    """q int64 [BN, G] permuted codes (16 per word), sc fp32 [BN, G] E4M3 block scale.
    -> DT [BN, G, 16] = e2m1 * scale * 2^-7, k-ordered."""
    s = sc * 6.646139978924579e+35  # 2^119
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
        w = _interleave_all(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)
        return w.to(DT)
    else:
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = tl.inline_asm_elementwise(
            _DEQ_ASM, "=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,l,h", [q, s.to(tl.bfloat16)],
            dtype=(tl.bfloat16,) * 16, is_pure=True, pack=1)
        return _interleave_all(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)


@triton.jit
def _moe_gemm(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, eids_ptr, npad_ptr,
              numel, local_start, N,
              K: tl.constexpr, A_ROW_DIV: tl.constexpr, GATED: tl.constexpr, ATOMIC: tl.constexpr,
              N_LOCAL: tl.constexpr, SPLIT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NATIVE: tl.constexpr):
    """out[route, n] for the routes of each sorted block. Weights per expert: int64 words
    [ROWS, K/16] of permuted codes (gate rows [0, N), up rows [N, 2N) when GATED), E4M3 block
    scales [ROWS, K/16] (one per word). A rows: a[route // A_ROW_DIV].
    ATOMIC: split-K partial sums, FP32 atomics into a zeroed FP32 buffer ([routes, 2N] gate | up
    when GATED). Otherwise (SPLIT == 1) a direct store: silu(gate) * up when GATED."""
    pid_n = tl.program_id(0)
    blk = tl.program_id(1)
    pid_k = tl.program_id(2)
    npad = tl.load(npad_ptr)
    e = tl.load(eids_ptr + blk)
    if (blk * BM >= npad) | (e < 0) | (e >= N_LOCAL):
        return
    G: tl.constexpr = BK // 16
    KW: tl.constexpr = K // 16
    KS: tl.constexpr = K // SPLIT
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_g = tl.arange(0, G)
    offs_k = tl.arange(0, BK)
    if GATED:
        rows_per_e = 2 * N
    else:
        rows_per_e = N
    routes = tl.load(sorted_ptr + blk * BM + tl.arange(0, BM))
    valid = routes < numel
    arow = tl.where(valid, routes // A_ROW_DIV, 0)
    eg = (e + local_start).to(tl.int64)
    k0 = pid_k * KS
    wbase = w_ptr + eg * rows_per_e * KW + k0 // 16
    sbase = sf_ptr + eg * rows_per_e * KW + k0 // 16
    x_ptrs = a_ptr + arow[None, :].to(tl.int64) * K + k0 + offs_k[:, None]
    w_off = offs_n[:, None] * KW + offs_g[None, :]
    acc_g = tl.zeros((BN, BM), dtype=tl.float32)
    acc_u = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in range(0, KS, BK):
        x = tl.load(x_ptrs + kk, mask=valid[None, :], other=0.0)
        q = tl.load(wbase + w_off + kk // 16)
        sc = tl.load(sbase + w_off + kk // 16).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        w = tl.reshape(_deq16(q, sc, x.dtype, NATIVE), (BN, BK))
        acc_g = tl.dot(w, x, acc_g)
        if GATED:
            q = tl.load(wbase + N * KW + w_off + kk // 16)
            sc = tl.load(sbase + N * KW + w_off + kk // 16).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            w = tl.reshape(_deq16(q, sc, x.dtype, NATIVE), (BN, BK))
            acc_u = tl.dot(w, x, acc_u)
    gs = tl.load(gs_ptr + eg) * 128.0  # undoes the 2^-7 of the dequant
    omask = valid[None, :] & (offs_n[:, None] < N)
    if ATOMIC:
        if GATED:
            o_ptrs = out_ptr + routes[None, :].to(tl.int64) * (2 * N) + offs_n[:, None]
            tl.atomic_add(o_ptrs, acc_g * gs, mask=omask, sem="relaxed")
            tl.atomic_add(o_ptrs + N, acc_u * gs, mask=omask, sem="relaxed")
        else:
            o_ptrs = out_ptr + routes[None, :].to(tl.int64) * N + offs_n[:, None]
            tl.atomic_add(o_ptrs, acc_g * gs, mask=omask, sem="relaxed")
    else:
        o_ptrs = out_ptr + routes[None, :].to(tl.int64) * N + offs_n[:, None]
        if GATED:
            g = acc_g * gs
            res = g * tl.sigmoid(g) * (acc_u * gs)
        else:
            res = acc_g * gs
        tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty), mask=omask)


@triton.jit
def _silu_mul_kernel(h_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """out[r, :] = silu(h[r, :N]) * h[r, N:] (FP32 math, stored in out's dtype)."""
    r = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(h_ptr + r * (2 * N) + offs)
    u = tl.load(h_ptr + r * (2 * N) + N + offs)
    tl.store(out_ptr + r * N + offs, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty))


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


# GEMM tile: BN=64 lets Triton use wgmma (64-row warpgroup tiles); BK=256 reads 128 B per row.
BN, BK, WARPS, STAGES = 64, 256, 4, 3
# EXPLORATION ONLY (check runs, not for submit): one tile config per scored token count, to
# learn several configs from one check. A submit uses one regime-level config.
SWEEP = {1: (64, 256, 4, 3), 8: (128, 128, 8, 3), 16: (64, 128, 4, 4), 32: (64, 256, 8, 3),
         128: (64, 128, 8, 4), 256: (64, 256, 4, 4)}
TARGET_CTAS = 4 * 132  # enough CTAs (bytes in flight) to stream weights at HBM speed


def _split(bs, n_tiles, k, top_k, count, n_experts, bk):
    """Split-K factor (power of two) from the expected number of local experts with work:
    a formula of the token count, so small batches (few experts) still fill the GPU."""
    routes = bs * top_k
    touched = count * (1.0 - (1.0 - 1.0 / n_experts) ** routes)
    want = TARGET_CTAS / max(1.0, touched * n_tiles)
    s = 1
    while s * 2 <= want and k // (s * 2) >= bk and s < 16:
        s *= 2
    return s


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


def _permute_bank(w, start, count, chunk=4):
    """Permute experts [start, start+count) only (the ones this GPU computes); the rest stay 0."""
    shp = w.shape
    out = torch.zeros((shp[0], shp[1], shp[2] // 8), dtype=torch.int64, device=w.device)
    for e in range(start, start + count, chunk):
        out[e:e + chunk] = permute_codes(w[e:min(e + chunk, start + count)])
    return out


def _align(local_ids, block_m, num_experts):
    try:
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    except ImportError:  # local CPU testing only
        return _align_torch(local_ids, block_m, num_experts)
    return moe_align_block_size(local_ids, block_m, num_experts)


def _align_torch(ids, block_m, num_experts):
    flat = ids.flatten().long()
    numel = flat.numel()
    maxp = numel + num_experts * (block_m - 1)
    sorted_ids = torch.full((maxp,), numel, dtype=torch.int32)
    eids = torch.full((maxp // block_m,), -1, dtype=torch.int32)
    pos = 0
    for e in range(num_experts):
        idx = (flat == e).nonzero().flatten()
        if idx.numel() == 0:
            continue
        nb = -(-idx.numel() // block_m)
        sorted_ids[pos:pos + idx.numel()] = idx.int()
        eids[pos // block_m: pos // block_m + nb] = e
        pos += nb * block_m
    return sorted_ids, eids, torch.tensor([pos], dtype=torch.int32)


def _post_norm_sum(y32, topk_weights, norm_w, slot_ids, out_dtype):
    bs, top_k, H = y32.shape
    out = torch.empty((bs, H), device=y32.device, dtype=out_dtype)
    _post_norm_sum_kernel[(bs,)](y32, norm_w, topk_weights, slot_ids, out, POST_NORM_EPS,
                                 TOPK=top_k, H=H, num_warps=8)
    return out


LOCAL = (0, 32)  # ep8: this GPU's experts


def prepare(weights):
    start, count = LOCAL
    st = {
        "w13": _permute_bank(weights["w13"], start, count),
        "w13_sf": weights["w13_sf"].contiguous().view(torch.uint8),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": _permute_bank(weights["w2"], start, count),
        "w2_sf": weights["w2_sf"].contiguous().view(torch.uint8),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "native": weights["w13"].device.type == "cpu",
    }
    if st["w13"].is_cuda:
        torch.cuda.synchronize()
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    bs, H = x.shape
    I = cfg["d_expert"]
    top_k = cfg["top_k"]
    start, count = cfg["local_start"], cfg["local_count"]
    assert (start, count) == LOCAL
    ids = data["topk_ids"].to(torch.int32) - start
    is_local = (ids >= 0) & (ids < count)
    slot_ids = torch.where(is_local, ids, -1).contiguous()
    bucket_ids = torch.where(is_local, ids, count).contiguous()  # non-local -> bucket `count`
    n_buckets = 64 if count < 64 else count + 1
    sorted_ids, eids, npad = _align(bucket_ids, BM, n_buckets)
    numel = bs * top_k
    max_blocks = min(eids.numel(), triton.cdiv(numel + (count + 1) * (BM - 1), BM))
    native = state["native"]
    n_exp = cfg["n_experts"]

    BN, BK, WARPS, STAGES = SWEEP[min(SWEEP, key=lambda b: abs(b - bs))]
    if native:
        BN, BK = 64, 256
    s1 = _split(bs, I // BN, H, top_k, count, n_exp, BK)
    s2 = _split(bs, H // BN, I, top_k, count, n_exp, BK)
    # One zeroed FP32 buffer for the split-K outputs: y = [routes, H] (+ h = [routes, 2I]).
    rows_h = numel if s1 > 1 else 0
    buf = torch.zeros((numel * H + rows_h * 2 * I,), device=x.device, dtype=torch.float32)
    y32 = buf[: numel * H].view(numel, H)
    inter = torch.empty((numel, I), device=x.device, dtype=x.dtype)
    if s1 > 1:
        h32 = buf[numel * H:].view(numel, 2 * I)
        _moe_gemm[(I // BN, max_blocks, s1)](
            x, state["w13"], state["w13_sf"], state["w13_gs"], h32, sorted_ids, eids, npad,
            numel, start, I, K=H, A_ROW_DIV=top_k, GATED=True, ATOMIC=True, N_LOCAL=count, SPLIT=s1,
            BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
        _silu_mul_kernel[(numel, I // 1024)](h32, inter, N=I, BLOCK=1024, num_warps=4)
    else:
        _moe_gemm[(I // BN, max_blocks, 1)](
            x, state["w13"], state["w13_sf"], state["w13_gs"], inter, sorted_ids, eids, npad,
            numel, start, I, K=H, A_ROW_DIV=top_k, GATED=True, ATOMIC=False, N_LOCAL=count, SPLIT=1,
            BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
    _moe_gemm[(H // BN, max_blocks, s2)](
        inter, state["w2"], state["w2_sf"], state["w2_gs"], y32, sorted_ids, eids, npad,
        numel, start, H, K=I, A_ROW_DIV=1, GATED=False, ATOMIC=True, N_LOCAL=count, SPLIT=s2,
        BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
    return _post_norm_sum(y32.view(bs, top_k, H), data["topk_weights"].float().contiguous(),
                          state["norm_w"], slot_ids, x.dtype)
