# okc: problem=muse_moe_w4a16
"""Routed experts (NVFP4 W4A16) as two Triton grouped GEMMs (swap-AB: weight rows are the
wgmma's M side, the <=16 routes of a sorted block its N side), dequantizing in registers
straight into the wgmma A operand (no shared-memory round trip of the BF16 weights).

- prepare() (lossless re-layouts, this GPU's 32 experts):
  * codes: within each 256-wide K block, byte j holds the codes of k = j (low nibble) and
    k = j + 128 (high nibble), so the two nibble halves of a byte tile are dotted against two
    contiguous halves of the activations;
  * block scales: E4M3 re-encoded exactly as BF16 (value * 2^119).
- Dequant, per 4 packed bytes (8 codes), in inline PTX: 2 prmt + 3-4 shift/and/lop3 per pair
  place each E2M1 code into a bf16 (value * 2^-126, exact, 0.5 is subnormal), then one
  mul.rn.bf16x2 by the block scale gives e2m1 * scale * 2^-7 exactly (<= 6 significant bits).
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

# $4 = 4 packed bytes b0..b3; $5,$6 = low-nibble scales (b0,b1),(b2,b3); $7,$8 = high-nibble
# scales. Outputs (bf16x2): $0,$1 = low-nibble values (b0,b1),(b2,b3); $2,$3 = high-nibble.
# Each output element depends only on its own input byte and scale (pairing-independent).
_DEQ_ASM = tl.constexpr("""
{
.reg .b32 u, v, t, r;
prmt.b32 u, $4, 0, 0x1404;
prmt.b32 v, $4, 0, 0x3424;
shr.b32 t, u, 6;
and.b32 t, t, 0x01C001C0;
lop3.b32 r, u, 0x80008000, t, 0xEA;
mul.rn.bf16x2 $2, r, $7;
shr.b32 t, v, 6;
and.b32 t, t, 0x01C001C0;
lop3.b32 r, v, 0x80008000, t, 0xEA;
mul.rn.bf16x2 $3, r, $8;
shl.b32 u, u, 4;
shr.b32 t, u, 6;
and.b32 t, t, 0x01C001C0;
lop3.b32 r, u, 0x80008000, t, 0xEA;
mul.rn.bf16x2 $0, r, $5;
shl.b32 v, v, 4;
shr.b32 t, v, 6;
and.b32 t, t, 0x01C001C0;
lop3.b32 r, v, 0x80008000, t, 0xEA;
mul.rn.bf16x2 $1, r, $6;
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
def _deq(q, s_lo, s_hi, NATIVE: tl.constexpr):
    """q uint8 [BN, BK/2]; s_lo / s_hi block scale * 2^119 (same shape) -> (low, high) nibble
    values e2m1 * scale * 2^-7."""
    if NATIVE:
        return _e2m1_native(q & 0xF) * s_lo, _e2m1_native(q >> 4) * s_hi
    else:
        return tl.inline_asm_elementwise(_DEQ_ASM, "=r,=r,=r,=r,r,r,r,r,r", [q, s_lo, s_hi],
                                         dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)


@triton.jit
def _moe_gemm(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, eids_ptr, npad_ptr,
              numel, N,
              K: tl.constexpr, A_ROW_DIV: tl.constexpr, GATED: tl.constexpr, ATOMIC: tl.constexpr,
              N_LOCAL: tl.constexpr, SPLIT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NATIVE: tl.constexpr):
    """out[route, n] for the routes of each sorted block. Weights per local expert: uint8
    [ROWS, K/2] re-laid codes (gate rows [0, N), up rows [N, 2N) when GATED), block scales
    [ROWS, K/16] (BF16 * 2^119). A rows: a[route // A_ROW_DIV].
    ATOMIC: split-K partial sums, FP32 atomics into a zeroed FP32 buffer ([routes, 2N] gate | up
    when GATED). Otherwise (SPLIT == 1) a direct store: silu(gate) * up when GATED."""
    pid_n = tl.program_id(0)
    blk = tl.program_id(1)
    pid_k = tl.program_id(2)
    npad = tl.load(npad_ptr)
    e = tl.load(eids_ptr + blk)
    if (blk * BM >= npad) | (e < 0) | (e >= N_LOCAL):
        return
    BH: tl.constexpr = BK // 2
    GH: tl.constexpr = BK // 32  # scale groups per nibble half
    KH: tl.constexpr = K // 2
    KS: tl.constexpr = K // 16
    KSPL: tl.constexpr = K // SPLIT
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_j = tl.arange(0, BH)
    offs_h = tl.arange(0, GH)
    offs_k = tl.arange(0, BH)
    if GATED:
        rows_per_e = 2 * N
    else:
        rows_per_e = N
    routes = tl.load(sorted_ptr + blk * BM + tl.arange(0, BM))
    valid = routes < numel
    arow = tl.where(valid, routes // A_ROW_DIV, 0)
    el = e.to(tl.int64)
    k0 = pid_k * KSPL
    w_ptrs = w_ptr + el * rows_per_e * KH + k0 // 2 + offs_n[:, None] * KH + offs_j[None, :]
    s_ptrs = sf_ptr + el * rows_per_e * KS + k0 // 16 + offs_n[:, None] * KS + offs_h[None, :]
    x_ptrs = a_ptr + arow[None, :].to(tl.int64) * K + k0 + offs_k[:, None]
    acc_g = tl.zeros((BN, BM), dtype=tl.float32)
    acc_u = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in range(0, KSPL, BK):
        x_lo = tl.load(x_ptrs + kk, mask=valid[None, :], other=0.0)
        x_hi = tl.load(x_ptrs + kk + BH, mask=valid[None, :], other=0.0)
        for half in tl.static_range(2 if GATED else 1):
            q = tl.load(w_ptrs + half * N * KH + kk // 2)
            sl = tl.load(s_ptrs + half * N * KS + kk // 16)
            sh = tl.load(s_ptrs + half * N * KS + kk // 16 + GH)
            if NATIVE:
                sl = sl.to(tl.float32)
                sh = sh.to(tl.float32)
            sl = tl.reshape(tl.broadcast_to(sl[:, :, None], (BN, GH, 16)), (BN, BH))
            sh = tl.reshape(tl.broadcast_to(sh[:, :, None], (BN, GH, 16)), (BN, BH))
            w_lo, w_hi = _deq(q, sl, sh, NATIVE)
            if half == 0:
                acc_g = tl.dot(w_lo.to(x_lo.dtype), x_lo, acc_g)
                acc_g = tl.dot(w_hi.to(x_lo.dtype), x_hi, acc_g)
            else:
                acc_u = tl.dot(w_lo.to(x_lo.dtype), x_lo, acc_u)
                acc_u = tl.dot(w_hi.to(x_lo.dtype), x_hi, acc_u)
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
# BN >= 64 so Triton uses wgmma with the dequantized A operand in registers.
SWEEP = {1: (64, 4, 3), 8: (64, 4, 4), 16: (128, 8, 3), 32: (64, 8, 3), 128: (128, 4, 3), 256: (64, 4, 2)}
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


def relayout_codes(w):
    """uint8 [..., K/2] packed E2M1 (low nibble = even k) -> uint8 [..., K/2] where, within each
    BKP-wide K block, byte j holds k = j (low nibble) and k = j + BKP/2 (high nibble)."""
    codes = torch.stack((w & 0xF, w >> 4), dim=-1).flatten(-2)  # [..., K], natural k order
    blocks = codes.unflatten(-1, (codes.shape[-1] // BKP, 2, BKP // 2))
    return (blocks[..., 0, :] | (blocks[..., 1, :] << 4)).flatten(-2).contiguous()


def _local_bank(w, start, count, chunk=4):
    return torch.cat([relayout_codes(w[e:min(e + chunk, start + count)])
                      for e in range(start, start + count, chunk)])


def _scales(sf, start, count, native):
    s = sf[start:start + count].float() * 2.0 ** 119  # exact; BF16 holds every E4M3 * 2^119
    return s.contiguous() if native else s.to(torch.bfloat16).contiguous()


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
        _moe_gemm[(I // BN, max_blocks, s1)](
            x, state["w13"], state["w13_sf"], state["w13_gs"], h32, sorted_ids, eids, npad,
            numel, I, K=H, A_ROW_DIV=top_k, GATED=True, ATOMIC=True, N_LOCAL=count, SPLIT=s1,
            BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
        blk_i = min(1024, I)
        _silu_mul_kernel[(numel, I // blk_i)](h32, inter, N=I, BLOCK=blk_i, num_warps=4)
    else:
        _moe_gemm[(I // BN, max_blocks, 1)](
            x, state["w13"], state["w13_sf"], state["w13_gs"], inter, sorted_ids, eids, npad,
            numel, I, K=H, A_ROW_DIV=top_k, GATED=True, ATOMIC=False, N_LOCAL=count, SPLIT=1,
            BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
    _moe_gemm[(H // BN, max_blocks, s2)](
        inter, state["w2"], state["w2_sf"], state["w2_gs"], y32, sorted_ids, eids, npad,
        numel, H, K=I, A_ROW_DIV=1, GATED=False, ATOMIC=True, N_LOCAL=count, SPLIT=s2,
        BM=BM, BN=BN, BK=BK, NATIVE=native, num_warps=WARPS, num_stages=STAGES)
    return _post_norm_sum(y32.view(bs, top_k, H), data["topk_weights"].float().contiguous(),
                          state["norm_w"], slot_ids, x.dtype)
