# okc: problem=muse_moe_w4a16
"""Routed experts (NVFP4 W4A16) as two Triton grouped GEMMs (swap-AB: weight rows are the
wgmma's M side, the <=16 routes of a sorted block its N side), dequantizing in registers
straight into the wgmma A operand (no shared-memory round trip of the BF16 weights).

- prepare() (lossless re-layouts, this GPU's 32 experts; every 64-row x 256-k tile of codes
  and of scales is stored contiguously, so a CTA streams whole chunks instead of 128-byte row pieces):
  * codes: within each 256-wide K block, 16-bit element j holds the codes of k = j, j + 64,
    j + 128, j + 192 (so the four code slots of a tile are dotted against four contiguous
    quarters of the activations), with the 16 bits permuted (_BITPOS) so that each code slot of
    an element pair (one 32-bit register) becomes a bf16x2 in 1-4 rotate/AND/LOP3 ops;
  * block scales: E4M3 re-encoded exactly as BF16 (value * 2^119).
- Dequant, per 32-bit register (two elements, 8 codes), in inline PTX: 11 shf/and/lop3 place
  each E2M1 code into a bf16 (value * 2^-126, exact, 0.5 is subnormal), then one
  mul.rn.bf16x2 per pair by the block scale gives e2m1 * scale * 2^-7 exactly (<= 6
  significant bits). Each output element depends only on its own input element and scale.
- BF16 operands, FP32 accumulation (wgmma); the epilogue multiplies by global scale * 2^7.
- Small token counts: split-K with FP32 atomics, then silu(gate) * up into the BF16
  intermediate (SGLang's rounding point); otherwise GEMM 1's epilogue does silu * up directly.
- An FP32 post-expert norm + weighted sum kernel combines the routes.
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
BKP = 256  # K block of the nibble re-layout (= the GEMM K tile)

# Bits of the four code slots inside a 16-bit element (mantissa, exponent lo, exponent hi, sign),
# found by a small search for the fewest ops: for an element pair in a 32-bit register r,
#   slot 0 = rotl(r, 6) & 0x81C081C0          slot 1 = r & 0x81C081C0
#   slot 2 = (rotl(r, 4) & 0x81808180) | (rotl(r, 28) & 0x00400040)
#   slot 3 = (rotl(r, 10) & 0x80008000) | (rotl(r, 26) & 0x01C001C0)
# (11 integer ops per 8 codes), putting each code's bits at bf16 bits 6, 7, 8 and 15.
_BITPOS = ((0, 1, 2, 9), (6, 7, 8, 15), (10, 3, 4, 11), (12, 13, 14, 5))

# $4 = elements (j, j+1); $5..$8 = slot 0..3 scales (bf16 * 2^119, low half used).
# Outputs $0..$3 = slot 0..3 values (bf16x2: element j low, j+1 high).
_DEQ_ASM = tl.constexpr("""
{
.reg .b32 r, t, u;
.reg .b16 h, x;
mov.b32 {h, x}, $5;
mov.b32 u, {h, h};
shf.l.wrap.b32 r, $4, $4, 6;
and.b32 r, r, 0x81C081C0;
mul.rn.bf16x2 $0, r, u;
mov.b32 {h, x}, $6;
mov.b32 u, {h, h};
and.b32 r, $4, 0x81C081C0;
mul.rn.bf16x2 $1, r, u;
mov.b32 {h, x}, $7;
mov.b32 u, {h, h};
shf.l.wrap.b32 t, $4, $4, 28;
and.b32 t, t, 0x00400040;
shf.l.wrap.b32 r, $4, $4, 4;
lop3.b32 r, r, 0x81808180, t, 0xEA;
mul.rn.bf16x2 $2, r, u;
mov.b32 {h, x}, $8;
mov.b32 u, {h, h};
shf.l.wrap.b32 t, $4, $4, 26;
and.b32 t, t, 0x01C001C0;
shf.l.wrap.b32 r, $4, $4, 10;
lop3.b32 r, r, 0x80008000, t, 0xEA;
mul.rn.bf16x2 $3, r, u;
}
""")


@triton.jit
def _e2m1_native(c):
    """CPU-interpreter twin: E2M1 code -> fp32 value * 2^-126 (what the bit placement gives)."""
    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * m, tl.exp2((e - 1).to(tl.float32)) * (1.0 + 0.5 * m))
    mag = tl.where((c & 8) != 0, -mag, mag)
    return mag * 1.1754943508222875e-38  # 2^-126


@triton.jit
def _slot_native(q, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr):
    """Read one code slot back out of the permuted 16-bit element (CPU interpreter only)."""
    c = ((q >> P0) & 1) | (((q >> P1) & 1) << 1) | (((q >> P2) & 1) << 2) | (((q >> P3) & 1) << 3)
    return _e2m1_native(c)


@triton.jit
def _deq(q, s0, s1, s2, s3, NATIVE: tl.constexpr):
    """q int16 [BN, BK/4] permuted elements; s0..s3 slot block scales * 2^119 (same shape)
    -> four slot value tensors e2m1 * scale * 2^-7."""
    if NATIVE:
        qi = q.to(tl.int32) & 0xFFFF
        return (_slot_native(qi, 0, 1, 2, 9) * s0, _slot_native(qi, 6, 7, 8, 15) * s1,
                _slot_native(qi, 10, 3, 4, 11) * s2, _slot_native(qi, 12, 13, 14, 5) * s3)
    else:
        return tl.inline_asm_elementwise(_DEQ_ASM, "=r,=r,=r,=r,r,r,r,r,r", [q, s0, s1, s2, s3],
                                         dtype=(tl.bfloat16,) * 4, is_pure=True, pack=2)


@triton.jit
def _bcast(s, BN: tl.constexpr, GQ: tl.constexpr, BQ: tl.constexpr, NATIVE: tl.constexpr):
    """[BN, GQ] block scales -> [BN, BQ] (each repeated over its 16 k)."""
    if NATIVE:
        s = s.to(tl.float32)
    return tl.reshape(tl.broadcast_to(s[:, :, None], (BN, GQ, 16)), (BN, BQ))


@triton.jit
def _work(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, blk, pid_n, pid_k, e, numel, N,
          K: tl.constexpr, A_ROW_DIV: tl.constexpr, GATED: tl.constexpr, ATOMIC: tl.constexpr,
          SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          NATIVE: tl.constexpr):
    """One work item: rows [pid_n * BN, +BN) x K slice pid_k of sorted block blk (expert e)."""
    BQ: tl.constexpr = BK // 4   # k per code slot per tile
    GQ: tl.constexpr = BK // 64  # scale groups per slot per tile
    KQ: tl.constexpr = K // 4
    KS: tl.constexpr = K // 16
    KSPL: tl.constexpr = K // SPLIT
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_j = tl.arange(0, BQ)
    offs_h = tl.arange(0, GQ)
    if GATED:
        rows_per_e = 2 * N
    else:
        rows_per_e = N
    routes = tl.load(sorted_ptr + blk * BM + tl.arange(0, BM))
    valid = routes < numel
    arow = tl.where(valid, routes // A_ROW_DIV, 0)
    el = e.to(tl.int64)
    k0 = pid_k * KSPL
    # Tiled layout: [row block of 64][K tile][64 rows][tile], so each tile is one contiguous chunk.
    TW: tl.constexpr = 64 * BQ        # elements per weight tile
    TS: tl.constexpr = 64 * (BK // 16)  # scales per tile
    rb = offs_n // 64
    rr = offs_n % 64
    w_ptrs = (w_ptr + el * rows_per_e * KQ + (k0 // BK) * TW
              + (rb * (64 * KQ) + rr * BQ)[:, None] + offs_j[None, :])
    s_ptrs = (sf_ptr + el * rows_per_e * KS + (k0 // BK) * TS
              + (rb * (64 * KS) + rr * (BK // 16))[:, None] + offs_h[None, :])
    x_ptrs = a_ptr + arow[None, :].to(tl.int64) * K + k0 + offs_j[:, None]
    acc_g = tl.zeros((BN, BM), dtype=tl.float32)
    acc_u = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in range(0, KSPL, BK):
        x0 = tl.load(x_ptrs + kk, mask=valid[None, :], other=0.0)
        x1 = tl.load(x_ptrs + kk + BQ, mask=valid[None, :], other=0.0)
        x2 = tl.load(x_ptrs + kk + 2 * BQ, mask=valid[None, :], other=0.0)
        x3 = tl.load(x_ptrs + kk + 3 * BQ, mask=valid[None, :], other=0.0)
        for half in tl.static_range(2 if GATED else 1):
            q = tl.load(w_ptrs + half * N * KQ + (kk // BK) * TW)
            sp = s_ptrs + half * N * KS + (kk // BK) * TS
            s0 = _bcast(tl.load(sp), BN, GQ, BQ, NATIVE)
            s1 = _bcast(tl.load(sp + GQ), BN, GQ, BQ, NATIVE)
            s2 = _bcast(tl.load(sp + 2 * GQ), BN, GQ, BQ, NATIVE)
            s3 = _bcast(tl.load(sp + 3 * GQ), BN, GQ, BQ, NATIVE)
            w0, w1, w2, w3 = _deq(q, s0, s1, s2, s3, NATIVE)
            if half == 0:
                acc_g = tl.dot(w0.to(x0.dtype), x0, acc_g)
                acc_g = tl.dot(w1.to(x0.dtype), x1, acc_g)
                acc_g = tl.dot(w2.to(x0.dtype), x2, acc_g)
                acc_g = tl.dot(w3.to(x0.dtype), x3, acc_g)
            else:
                acc_u = tl.dot(w0.to(x0.dtype), x0, acc_u)
                acc_u = tl.dot(w1.to(x0.dtype), x1, acc_u)
                acc_u = tl.dot(w2.to(x0.dtype), x2, acc_u)
                acc_u = tl.dot(w3.to(x0.dtype), x3, acc_u)
    gs = tl.load(gs_ptr + el) * 128.0  # undoes the 2^-7 of the dequant
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
def _moe_gemm(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, eids_ptr, npad_ptr,
              numel, N,
              K: tl.constexpr, A_ROW_DIV: tl.constexpr, GATED: tl.constexpr, ATOMIC: tl.constexpr,
              N_LOCAL: tl.constexpr, SPLIT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NATIVE: tl.constexpr):
    """out[route, n] for the routes of each sorted block. Weights per local expert: int16
    [ROWS, K/4] re-laid codes (gate rows [0, N), up rows [N, 2N) when GATED), block scales
    [ROWS, K/16] (BF16 * 2^119). A rows: a[route // A_ROW_DIV].
    ATOMIC: split-K partial sums, FP32 atomics into a zeroed FP32 buffer ([routes, 2N] gate | up
    when GATED). Otherwise (SPLIT == 1) a direct store: silu(gate) * up when GATED.
    Persistent: each program walks work items (block, row tile, K slice) with a stride of the
    grid size, over only the blocks the routing produced (no empty programs to launch)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    npad = tl.load(npad_ptr)
    per_blk = (N // BN) * SPLIT
    n_work = (npad // BM) * per_blk
    for w in range(pid, n_work, nprog):
        blk = w // per_blk
        rem = w % per_blk
        e = tl.load(eids_ptr + blk)
        if (e >= 0) & (e < N_LOCAL):
            _work(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, blk, rem // SPLIT, rem % SPLIT, e,
                  numel, N, K, A_ROW_DIV, GATED, ATOMIC, SPLIT, BM, BN, BK, NATIVE)


@triton.jit
def _route_prep(ids_ptr, slot_ptr, bucket_ptr, n, start, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """slot = local expert id or -1; bucket = local expert id, or COUNT for routes to other GPUs."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    i = tl.load(ids_ptr + offs, mask=m, other=0).to(tl.int32) - start
    loc = (i >= 0) & (i < COUNT)
    tl.store(slot_ptr + offs, tl.where(loc, i, -1), mask=m)
    tl.store(bucket_ptr + offs, tl.where(loc, i, COUNT), mask=m)


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
# EXPLORATION ONLY (check runs, not for submit): one (BN, num_warps, num_stages) per scored
# token count, to learn several configs from one check. A submit uses one regime-level config.
# BN=32/2 warps (or 64/2) gives mma.sync with the dequantized A operand in registers and no
# per-dot wgmma wait + fence (BN=64 / 4 warps = wgmma, kept at 128 tokens as a control).
SWEEP = {1: (32, 2, 2), 8: (32, 1, 2), 16: (32, 2, 2), 32: (32, 2, 3), 128: (32, 1, 3), 256: (32, 2, 2)}
TARGET_CTAS = 6 * 132  # resident CTA slots (BN=32, 2 warps: ~6 per SM by registers / shared memory)
PERSISTENT_CTAS = 32 * 132  # residency is left to the hardware (registers / shared memory)


def _grid(max_work):
    return max(1, min(max_work, PERSISTENT_CTAS))


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


def relayout_codes(w):
    """uint8 [..., K/2] packed E2M1 (low nibble = even k) -> int16 [..., K/4]: within each
    BKP-wide K block, element j holds the codes of k = j + i * BKP/4 in slot i (i = 0..3),
    each code's 4 bits moved to _BITPOS[i] (a lossless bit permutation)."""
    codes = torch.stack((w & 0xF, w >> 4), dim=-1).flatten(-2).to(torch.int32)  # natural k order
    blocks = codes.unflatten(-1, (codes.shape[-1] // BKP, 4, BKP // 4))
    out = torch.zeros_like(blocks[..., 0, :])
    for i, pos in enumerate(_BITPOS):
        for b, p in enumerate(pos):
            out |= ((blocks[..., i, :] >> b) & 1) << p
    out = torch.where(out >= 2 ** 15, out - 2 ** 16, out)
    return out.flatten(-2).to(torch.int16).contiguous()


def _tile(t, k_per_tile):
    """[E, ROWS, KX] -> same bytes reordered as [E, ROWS/64, KX/k_per_tile, 64, k_per_tile]."""
    e, rows, kx = t.shape
    return t.view(e, rows // 64, 64, kx // k_per_tile, k_per_tile).permute(0, 1, 3, 2, 4).contiguous().view(e, rows, kx)


def _local_bank(w, start, count, chunk=4):
    return torch.cat([_tile(relayout_codes(w[e:min(e + chunk, start + count)]), BKP // 4)
                      for e in range(start, start + count, chunk)])


def _scales(sf, start, count, native):
    s = sf[start:start + count].float() * 2.0 ** 119  # exact; BF16 holds every E4M3 * 2^119
    s = s if native else s.to(torch.bfloat16)
    return _tile(s.contiguous(), BKP // 16)


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
    native = weights["w13"].device.type == "cpu"
    st = {
        "w13": _local_bank(weights["w13"], start, count),
        "w13_sf": _scales(weights["w13_sf"], start, count, native),
        "w13_gs": weights["w13_gs"][start:start + count].float().contiguous(),
        "w2": _local_bank(weights["w2"], start, count),
        "w2_sf": _scales(weights["w2_sf"], start, count, native),
        "w2_gs": weights["w2_gs"][start:start + count].float().contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "native": native,
    }
    if not native:
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
    ids = data["topk_ids"].contiguous()
    slot_ids = torch.empty(ids.shape, device=ids.device, dtype=torch.int32)
    bucket_ids = torch.empty(ids.shape, device=ids.device, dtype=torch.int32)  # non-local -> `count`
    _route_prep[(triton.cdiv(ids.numel(), 1024),)](ids, slot_ids, bucket_ids, ids.numel(), start,
                                                     COUNT=count, BLOCK=1024)
    n_buckets = 64 if count < 64 else count + 1
    sorted_ids, eids, npad = _align(bucket_ids, BM, n_buckets)
    numel = bs * top_k
    max_blocks = min(eids.numel(), triton.cdiv(numel + (count + 1) * (BM - 1), BM))
    # The local blocks are at most one partly filled block per touched expert plus full ones.
    max_local_blocks = min(max_blocks, numel // BM + min(numel, count) + 1)
    native = state["native"]
    n_exp = cfg["n_experts"]

    BN, WARPS, STAGES = SWEEP[min(SWEEP, key=lambda b: abs(b - bs))]
    BK = BKP
    s1 = _split(bs, I // BN, H, top_k, count, n_exp, BK)
    s2 = _split(bs, H // BN, I, top_k, count, n_exp, BK)
    # One zeroed FP32 buffer for the split-K outputs: y = [routes, H] (+ h = [routes, 2I]).
    rows_h = numel if s1 > 1 else 0
    buf = torch.zeros((numel * H + rows_h * 2 * I,), device=x.device, dtype=torch.float32)
    y32 = buf[: numel * H].view(numel, H)
    inter = torch.empty((numel, I), device=x.device, dtype=x.dtype)
    if s1 > 1:
        h32 = buf[numel * H:].view(numel, 2 * I)
        _moe_gemm[(_grid(I // BN * s1 * max_local_blocks),)](
            x, state["w13"], state["w13_sf"], state["w13_gs"], h32, sorted_ids, eids, npad,
            numel, I, K=H, A_ROW_DIV=top_k, GATED=True, ATOMIC=True, N_LOCAL=count, SPLIT=s1,
            BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
        blk_i = min(1024, I)
        _silu_mul_kernel[(numel, I // blk_i)](h32, inter, N=I, BLOCK=blk_i, num_warps=4)
    else:
        _moe_gemm[(_grid(I // BN * max_local_blocks),)](
            x, state["w13"], state["w13_sf"], state["w13_gs"], inter, sorted_ids, eids, npad,
            numel, I, K=H, A_ROW_DIV=top_k, GATED=True, ATOMIC=False, N_LOCAL=count, SPLIT=1,
            BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
    _moe_gemm[(_grid(H // BN * s2 * max_local_blocks),)](
        inter, state["w2"], state["w2_sf"], state["w2_gs"], y32, sorted_ids, eids, npad,
        numel, H, K=I, A_ROW_DIV=1, GATED=False, ATOMIC=True, N_LOCAL=count, SPLIT=s2,
        BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
    return _post_norm_sum(y32.view(bs, top_k, H), data["topk_weights"].float().contiguous(),
                          state["norm_w"], slot_ids, x.dtype)
