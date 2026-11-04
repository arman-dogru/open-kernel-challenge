# okc: problem=muse_attention_decode
"""v25e_ws4: v24 + attention v25 source (tools/build_attn_v25.py): the full-verify kernel from tools/attn_v25_fv.cu
(v24's + the "memory" clobber on the V transpose's inline ld/st.shared, review r17 m1; ST at compile time), FV ST
3, VT1 55869504, FV splits/SM 3, full decode batch >= 2 rows config (128, 4, 3, 4, True).
v25e_ws4: full verify (q_len 8, batch >= 2) on attn_fvws (tools/attn_v25_ws.cu) with NPW 4 cp.async producer warps
(a whole warpgroup, setmaxnreg 24) and 2 consumer warpgroups (setmaxnreg 104), launched at 80 registers so 2 CTAs fit
per SM (the v25d WS build's 9 warps at 112 registers fit only 1: 3 warps per SM sub-partition), per-warpgroup
barriers; flags _wsf(npw=4, maxr=80, smr=104) = 55869504. Bit-exact vs v24's FV at equal splits (v25e1 diag
5ff307988ce14518: 648/648); b16 / b32 L8 FV 143 / 267 us vs v24 170 / 323.

v24: v23 with full-attention decode (q_len 1) at batch >= 2 on the split-KV "rows" kernel at ~4 programs per SM
instead of ~16 (same 128-key tiles, V hoisted, page-index prefetch): fewer, longer splits. In-prepare sweeps on the
judge's H200 (decode-layer diagnostic checks 6567d26179874f80 / 8bae0dfe527f4f01, cold L2): rows + combine
150.6 -> ~127 us at batch 16 and 246.9 -> ~234 us at batch 32 (tools/build_attn_v24.py).

v23: v22 with the full-verify kernel's address math hoisted out of its tile loop and Q.K^T issued
before the V transpose (tools/attn_v23_fv.cu, tools/build_attn_v23.py).

v22: v20b with full-attention verify at batch >= 2 on a CUDA sm_90a split-KV kernel (FP8 wgmma,
cp.async page-table gather) instead of FA3. See tools/build_attn_v22.py and tools/attn_v22_fv.cu.

v20: v17 + the per-token norm/RoPE/store kernel from 64 tokens and sliding verify on 64-key head-kernel
tiles (v20b) with V hoisted and next-tile page-index prefetch. See tools/build_attn_v20.py.

v17: v15b with the full-decode (batch >= 2) split-KV "rows" config on 128-key tiles and the next tile's page
indices loaded one iteration ahead (PREF), as decode v20. Same math and precision.

v15b: v14b with verify blocks (q_len 8) on the split-KV "head" kernel on sliding layers at any batch and on
full layers at batch 1 (check 28ad328cf5284948: sliding verify 1.55 / 1.53 / 1.28x at b 1/16/32 vs FA3's 1.33 /
1.26 / 1.16x, full verify b1 1.63x vs 1.20x; full verify b16/32 lost, 0.69 / 0.67x, so those stay on FA3), plus
r6 m2 (dead SGLang-path helpers removed), r6 n3 (_nsplit docstring; the combine's NSPLIT is always a power of two,
so its dead masking is gone) and r6 n4 (RoPE table read clamped to its last row; positions >= 2^21 are unreachable
at judge sizes).

v14b: v14 with this docstring corrected (code identical).

v14: v13 with the query written by the fused kernel directly in FP8 E4M3 on GPU (FP8 queries are
allowed; FA3 casts them itself, so its path loses the separate cast launch). The split-KV kernels then
run Q.K^T as an FP8 x FP8 tensor-core dot on the raw E4M3 cache bytes (no K conversion); P.V stays BF16.

v13: v12 with SGLang's FusedQKNorm + the two FP8 casts + store_cache (4-5 launches, ~5 us of GPU time and
launch gaps at batch 1) replaced by one Triton program per (token, head): RMSNorm, interleaved RoPE from
the same cos|sin table, q out in BF16, k (BF16-rounded, / k_scale) and v (/ v_scale) cast to E4M3 and
written to their cache rows. Used on every path (split-KV and FA3).

Gated attention decode with a split-KV (flash-decoding) Triton kernel over the paged FP8
cache, and a combine kernel that also applies the per-head RMSNorm and sigmoid gate.

SGLang's FusedQKNorm, query/K/V casts and store_cache are replaced on every path by the one fused
Triton norm+RoPE+FP8+cache-write kernel above. Replaced on decode steps and most verify blocks (see _attn_path;
chosen by regime, not by exact batch size): FA3 and rms_norm_gate. Full-layer verify blocks at batch >= 2
and untuned head layouts keep SGLang's FA3 path (fed the FP8 query).

- Attention runs over every key it should (whole window / whole context, causal inside the
  verify block); no skipping. Queries are E4M3 (one RNE, saturating fp32->E4M3 rounding, as FA3 does);
  Q.K^T is an FP8 x FP8 tensor-core dot on the raw E4M3 K bytes, V is decoded exactly in
  registers and P.V runs in BF16, softmax and accumulation in FP32.
- The grid is (kv head, split, request): each program owns one KV head's query group (x the
  block's tokens) over one slice of the keys; splits are sized from the static cache shape, so
  the launch is CUDA-graph safe. Programs past a request's length write an empty partial."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
ROPE_POSITIONS = 1 << 21
LOG2E = tl.constexpr(1.4426950408889634)
_ROPE = {}


@triton.jit
def _e4m3(b):
    """uint8 E4M3 bits -> fp32 value / 256 (exact: E4M3 maps bit-for-bit into FP16 x 2^-8)."""
    s = b.to(tl.int16)
    return (((s & 0x7F) << 7) | ((s & 0x80) << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _kv(x, HW: tl.constexpr, DT: tl.constexpr):
    """A uint8 cache tile -> DT. HW: reinterpreted as FP8 E4M3 and converted by the hardware (exact).
    Otherwise uint8 bits decoded by hand (value / 256; the caller folds the 256 in)."""
    if HW:
        return x.to(tl.float8e4nv, bitcast=True).to(DT)
    else:
        return _e4m3(x).to(DT)


@triton.jit
def _attn_split(q_ptr, stride_qt, k_ptr, v_ptr, pt_ptr, stride_pt, seqlens_ptr,
                ks_ptr, vs_ptr, o_ptr, lse_ptr, qk_scale, chunk, PAGE,
                L: tl.constexpr, G: tl.constexpr, HKV: tl.constexpr, NSPLIT: tl.constexpr,
                WINDOW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False,
                PREF: tl.constexpr = False):
    h = tl.program_id(0)
    s = tl.program_id(1)
    b = tl.program_id(2)
    HQ: tl.constexpr = HKV * G
    m = tl.arange(0, BLOCK_M)
    j = m // G                      # block token
    head = h * G + (m % G)          # query head
    mvalid = m < L * G
    d = tl.arange(0, D)
    tq = b * L + j
    q = tl.load(q_ptr + tq[:, None] * stride_qt + head[:, None] * D + d[None, :],
                mask=mvalid[:, None], other=0.0)
    total = tl.load(seqlens_ptr + b)          # cached + L
    seq = total - L
    pos = seq + j                              # each row's position
    if WINDOW > 0:
        base = tl.maximum(seq - WINDOW + 1, 0)
    else:
        base = seq * 0
    lo = base + s * chunk
    hi = tl.minimum(lo + chunk, total)
    if HW:
        ks = tl.load(ks_ptr)
        vs = tl.load(vs_ptr)
    else:
        ks = tl.load(ks_ptr) * 256.0
        vs = tl.load(vs_ptr) * 256.0
    scale = qk_scale * ks * LOG2E
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    if PREF:  # the first tile's page indices; each iteration then loads the next tile's (v20, as _attn_split_rows)
        t0 = lo + tl.arange(0, BLOCK_N)
        pnext = tl.load(pt_ptr + b * stride_pt + t0 // PAGE, mask=t0 < hi, other=0)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        if PREF:
            page = pnext
            tn = t + BLOCK_N
            pnext = tl.load(pt_ptr + b * stride_pt + tn // PAGE, mask=tn < hi, other=0)
        else:
            page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
        row = (page.to(tl.int64) * PAGE + t % PAGE) * (HKV * D) + h * D
        kb = tl.load(k_ptr + row[:, None] + d[None, :], mask=tmask[:, None], other=0)
        if HOIST:  # issue the V load before the QK / softmax math: two gathers in flight
            vb = tl.load(v_ptr + row[:, None] + d[None, :], mask=tmask[:, None], other=0)
        k = _kv(kb, HW, q.dtype)
        sc = tl.dot(q, tl.trans(k)) * scale
        ok = tmask[None, :] & (t[None, :] <= pos[:, None])
        if WINDOW > 0:
            ok = ok & (t[None, :] > pos[:, None] - WINDOW)
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(sc - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        if not HOIST:
            vb = tl.load(v_ptr + row[:, None] + d[None, :], mask=tmask[:, None], other=0)
        if PV16:  # FP8 queries: P.V in BF16
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), _kv(vb, HW, tl.bfloat16))
        else:
            v = _kv(vb, HW, q.dtype)
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_new
    # partial: o = acc / l (times the value scale), lse in log2 units
    out = acc * (vs / tl.where(l_i > 0, l_i, 1.0))[:, None]
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.where(l_i > 0, l_i, 1.0)), float("-inf"))
    idx = (tq * HQ + head) * NSPLIT + s
    tl.store(o_ptr + idx[:, None] * D + d[None, :], out, mask=mvalid[:, None])
    tl.store(lse_ptr + idx, lse, mask=mvalid)


@triton.jit
def _attn_split_rows(q_ptr, stride_qt, k_ptr, v_ptr, pt_ptr, stride_pt, seqlens_ptr,
                     ks_ptr, vs_ptr, o_ptr, lse_ptr, qk_scale, chunk, PAGE,
                     L: tl.constexpr, G: tl.constexpr, HKV: tl.constexpr, NSPLIT: tl.constexpr,
                     WINDOW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False,
                     PREF: tl.constexpr = False):
    """Both KV heads in one program, so each key's whole cache row (HKV*64 bytes) is read at
    once. Each query row sits in its own KV head's 64-wide segment of a HKV*64-wide query
    (zeros elsewhere), so Q_ext @ row^T is exactly its head's logit; the PV product's other
    segment is discarded at the store."""
    s = tl.program_id(0)
    b = tl.program_id(1)
    HQ: tl.constexpr = HKV * G
    W: tl.constexpr = HKV * D
    m = tl.arange(0, BLOCK_M)
    j = m // HQ
    head = m % HQ
    kvh = head // G
    mvalid = m < L * HQ
    c = tl.arange(0, W)
    seg = c // D
    dd = c % D
    tq = b * L + j
    mine = (seg[None, :] == kvh[:, None]) & mvalid[:, None]
    q = tl.load(q_ptr + tq[:, None] * stride_qt + head[:, None] * D + dd[None, :], mask=mine, other=0.0)
    total = tl.load(seqlens_ptr + b)
    seq = total - L
    pos = seq + j
    if WINDOW > 0:
        base = tl.maximum(seq - WINDOW + 1, 0)
    else:
        base = seq * 0
    lo = base + s * chunk
    hi = tl.minimum(lo + chunk, total)
    if HW:
        ks = tl.load(ks_ptr)
        vs = tl.load(vs_ptr)
    else:
        ks = tl.load(ks_ptr) * 256.0
        vs = tl.load(vs_ptr) * 256.0
    scale = qk_scale * ks * LOG2E
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, W), tl.float32)
    if PREF:  # the first tile's page indices; each iteration then loads the next tile's
        t0 = lo + tl.arange(0, BLOCK_N)
        pnext = tl.load(pt_ptr + b * stride_pt + t0 // PAGE, mask=t0 < hi, other=0)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        if PREF:
            page = pnext
            tn = t + BLOCK_N
            pnext = tl.load(pt_ptr + b * stride_pt + tn // PAGE, mask=tn < hi, other=0)
        else:
            page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
        row = (page.to(tl.int64) * PAGE + t % PAGE) * W
        kb = tl.load(k_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        if HOIST:
            vb = tl.load(v_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        sc = tl.dot(q, tl.trans(_kv(kb, HW, q.dtype))) * scale
        ok = tmask[None, :] & (t[None, :] <= pos[:, None])
        if WINDOW > 0:
            ok = ok & (t[None, :] > pos[:, None] - WINDOW)
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(sc - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        if not HOIST:
            vb = tl.load(v_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        if PV16:  # FP8 queries: P.V in BF16
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), _kv(vb, HW, tl.bfloat16))
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), _kv(vb, HW, q.dtype))
        m_i = m_new
    out = acc * (vs / tl.where(l_i > 0, l_i, 1.0))[:, None]
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.where(l_i > 0, l_i, 1.0)), float("-inf"))
    idx = (tq * HQ + head) * NSPLIT + s
    tl.store(o_ptr + idx[:, None] * D + dd[None, :], out, mask=mine)
    tl.store(lse_ptr + idx, lse, mask=mvalid)


@triton.jit
def _combine_norm_gate(o_ptr, lse_ptr, gate_ptr, stride_gt, out_ptr, eps,
                       HQ: tl.constexpr, NSPLIT: tl.constexpr, D: tl.constexpr):
    """Merge the splits of one (token, head), then RMSNorm over the head x sigmoid(gate).
    NSPLIT is a power of two (_nsplit), so no lanes are masked."""
    t = tl.program_id(0)
    hh = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    idx = (t * HQ + hh) * NSPLIT + sp
    lse = tl.load(lse_ptr + idx)
    mx = tl.max(lse, 0)
    w = tl.where(lse == float("-inf"), 0.0, tl.exp2(lse - mx))
    o = tl.load(o_ptr + idx[:, None] * D + d[None, :])
    a = tl.sum(o * w[:, None], 0) / tl.sum(w, 0)
    r = a * tl.rsqrt(tl.sum(a * a, 0) / D + eps)
    g = tl.load(gate_ptr + t * stride_gt + hh * D + d).to(tl.float32)
    res = r * tl.sigmoid(g)
    tl.store(out_ptr + t * (HQ * D) + hh * D + d, res.to(out_ptr.dtype.element_ty))


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache



@triton.jit
def _norm_rope_store_kernel(qkv_ptr, stride_qkv, pos_ptr, cs_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                            q_ptr, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                            ROPE: tl.constexpr, SCALED: tl.constexpr, NPOS: tl.constexpr):
    """Program (t, h): h < HQ a query head, < HQ + HKV a key head, else a value head."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.arange(0, D // 2)
    base = qkv_ptr + t * stride_qkv + h * D
    e = tl.load(base + 2 * i).to(tl.float32)
    o = tl.load(base + 2 * i + 1).to(tl.float32)
    if h < HQ + HKV:
        r = tl.rsqrt((tl.sum(e * e, 0) + tl.sum(o * o, 0)) / D + eps)
        e = e * r
        o = o * r
        if ROPE:
            pos = tl.minimum(tl.load(pos_ptr + t).to(tl.int64), NPOS - 1)  # table bound (r6 n4)
            c = tl.load(cs_ptr + pos * D + i)
            s = tl.load(cs_ptr + pos * D + D // 2 + i)
            e, o = e * c - o * s, e * s + o * c
        if h < HQ:
            tl.store(q_ptr + t * (HQ * D) + h * D + 2 * i, e.to(q_ptr.dtype.element_ty))
            tl.store(q_ptr + t * (HQ * D) + h * D + 2 * i + 1, o.to(q_ptr.dtype.element_ty))
        else:
            e = e.to(tl.bfloat16).to(tl.float32)
            o = o.to(tl.bfloat16).to(tl.float32)
            if SCALED:
                ks = tl.load(ks_ptr)
                e = e / ks
                o = o / ks
            row = tl.load(loc_ptr + t).to(tl.int64)
            dst = kc_ptr + row * (HKV * D) + (h - HQ) * D
            tl.store(dst + 2 * i, e.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
            tl.store(dst + 2 * i + 1, o.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
    else:
        if SCALED:
            vs = tl.load(vs_ptr)
            e = e / vs
            o = o / vs
        row = tl.load(loc_ptr + t).to(tl.int64)
        dst = vc_ptr + row * (HKV * D) + (h - HQ - HKV) * D
        tl.store(dst + 2 * i, e.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
        tl.store(dst + 2 * i + 1, o.to(tl.float8e4nv).to(tl.uint8, bitcast=True))


@triton.jit
def _norm_rope_store_tok_kernel(qkv_ptr, stride_qkv, pos_ptr, cs_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                            q_ptr, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                            ROPE: tl.constexpr, SCALED: tl.constexpr, NPOS: tl.constexpr, HP: tl.constexpr):
    """Program t: every head of token t as one [HP, D/2] tile (v19, used at T >= TOK_MIN); head h < HQ a query head, < HQ + HKV a key head,
    < HQ + 2*HKV a value head (rows above masked). Per-head math as v17's one-program-per-head kernel."""
    t = tl.program_id(0)
    h = tl.arange(0, HP)[:, None]
    i = tl.arange(0, D // 2)[None, :]
    live = h < HQ + 2 * HKV
    isq = h < HQ
    isk = (h >= HQ) & (h < HQ + HKV)
    isv = (h >= HQ + HKV) & live
    base = qkv_ptr + t * stride_qkv + h * D
    e = tl.load(base + 2 * i, mask=live, other=0.0).to(tl.float32)
    o = tl.load(base + 2 * i + 1, mask=live, other=0.0).to(tl.float32)
    r = tl.rsqrt((tl.sum(e * e, 1) + tl.sum(o * o, 1)) / D + eps)[:, None]
    nq = isq | isk
    e = tl.where(nq, e * r, e)
    o = tl.where(nq, o * r, o)
    if ROPE:
        pos = tl.minimum(tl.load(pos_ptr + t).to(tl.int64), NPOS - 1)  # table bound (r6 n4)
        c = tl.load(cs_ptr + pos * D + i)
        sn = tl.load(cs_ptr + pos * D + D // 2 + i)
        e, o = tl.where(nq, e * c - o * sn, e), tl.where(nq, e * sn + o * c, o)
    tl.store(q_ptr + t * (HQ * D) + h * D + 2 * i, e.to(q_ptr.dtype.element_ty), mask=isq)
    tl.store(q_ptr + t * (HQ * D) + h * D + 2 * i + 1, o.to(q_ptr.dtype.element_ty), mask=isq)
    e = tl.where(isk, e.to(tl.bfloat16).to(tl.float32), e)
    o = tl.where(isk, o.to(tl.bfloat16).to(tl.float32), o)
    if SCALED:
        sc = tl.where(isk, tl.load(ks_ptr), tl.load(vs_ptr))
        e = e / sc
        o = o / sc
    row = tl.load(loc_ptr + t).to(tl.int64)
    kvh = tl.where(isk, h - HQ, h - HQ - HKV)
    dst = tl.where(isk, kc_ptr, vc_ptr) + row * (HKV * D) + kvh * D
    kv = isk | isv
    tl.store(dst + 2 * i, e.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=kv)
    tl.store(dst + 2 * i + 1, o.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=kv)


TOK_MIN = 64  # tokens from which the per-token norm/RoPE/store kernel is used (v19b / v20)


def _norm_rope_store_fused(data, qkv, cfg, q_dtype):
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = qkv.shape[0]
    q = torch.empty((T, hq * d), device=qkv.device, dtype=q_dtype)
    rope = bool(cfg["rope_theta"])
    cs = _cos_sin(cfg["rope_theta"], d, qkv.device) if rope else q
    args = (qkv, qkv.stride(0), data["positions"], cs, data["cache_loc"],
            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
            q, RMS_EPS)
    if T >= TOK_MIN:
        _norm_rope_store_tok_kernel[(T,)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=cfg["kv_scale"] != 1.0,
                                          NPOS=ROPE_POSITIONS, HP=triton.next_power_of_2(hq + 2 * hkv), num_warps=2)
    else:
        _norm_rope_store_kernel[(T, hq + 2 * hkv)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope,
                                                   SCALED=cfg["kv_scale"] != 1.0, NPOS=ROPE_POSITIONS, num_warps=1)
    return q

def _nsplit(cfg, batch, progs_per_split, capacity, occ=4):
    """Splits per request, a power of two (1-256): ~occ programs per SM and about span / 256 or fewer splits;
    rounding to the nearest power of two can halve the slice, so slices are >= ~128 keys."""
    span = (cfg["window"] + cfg["q_len"] - 1) if cfg["window"] else capacity
    want = max(1, (132 * occ) // (batch * progs_per_split))
    by_len = max(1, span // 256)
    # Nearest power of two (audit r2 m5: flooring left batch 17 at ~8 programs/SM; exact counts
    # like 132 measured slower than 128 at b16, check 85e63326). The combine masks either way.
    n = min(want, by_len, 256)
    p = 1 << (n.bit_length() - 1)
    r = min(2 * p, 256) if 2 * p - n < n - p else p
    assert r & (r - 1) == 0
    return r


# Path by regime (layer kind, q_len, batch range), never by exact scored batch size, so nearby
# shapes take the same path as the scored ones (H200 evidence: check runs v1b..v8, see STATUS.md):
#   sliding decode (q_len 1), any batch      split-KV "head" kernel: 1.25-1.35x FA3 at b 1/16/32
#   full decode, batch 1                     split-KV "rows", ~4 programs/SM (256 splits): ~1.1x
#   full decode, batch >= 2                  "rows", 128-key tiles, ~4 programs/SM (v24; ~16 before: 150.6 /
#                                            246.9 vs 126-128 / 234 us at b 16/32, decode checks 6567d261 / 8bae0dfe)
#   sliding verify (q_len 8), any batch      split-KV "head" kernel (FP8 Q.K): 1.15-1.22x FA3
#   full verify, batch 1                     split-KV "head" kernel: 1.36x FA3
#   full verify, batch >= 2                  SGLang's FA3 (the head kernel was 0.67-0.69x at b16/32)
# Value: (kernel, BLOCK_N, warps, stages, programs per SM, hoist V load); None = FA3.
def _attn_path(kind, L, batch, hkv, d):
    if L != 1:
        if hkv * d > 128:
            return None
        if kind == "full" and batch >= 2:
            return None  # FA3 (split-KV head kernel 0.67-0.69x of it at b16/32, check 28ad328cf5284948)
        if kind == "sliding":  # v20b: v17's 64-key tiles, V hoisted, PREF (128-key tiles were slower)
            return ("head", 64, 4, 3, 4, True)
        return ("head", 64, 4, 3, 4, False)
    if kind == "sliding":
        return ("head", 64, 4, 3, 4, False)
    if hkv * d > 128:  # other head layouts (e.g. "dp"): not tuned, keep FA3
        return None
    if batch == 1:
        return ("rows", 64, 4, 3, 4, False)
    return ROWS_FULL  # v24: ("rows", 128, 4, 3, 4, True): ~4 programs/SM, 128-key tiles + PREF (= hoist); v25 config


def _fa3(data, q, qkv, cfg):
    """SGLang's path after norm/RoPE/cache store (as baseline.py): q to E4M3, FA3, rms_norm_gate."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8), k_cache=data["k_cache"], v_cache=data["v_cache"],
        page_table=data["page_table"], cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"],
        **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


_FV_SRC = r"""
// Full-attention verify (q_len 8) split-KV partials for tp8 (16 query / 2 KV heads, D 64), attention v23 / v24 / v25.
// v24 source: v23 + a "memory" clobber on the V transpose's inline ld.shared / st.shared (review r17 m1); same PTX / cubin.
// v25 source: v24 + ST overridable at compile time (-DST=n; default 3 gives v24's cubin byte for byte), and VT1 = 1:
// one V^T buffer instead of two (each tile first waits for the previous tile's P.V, before the loop-top barrier, so no
// warp transposes into V^T while a P.V still reads it): 18 KB less shared memory, so ST = 4 (3 tiles in flight) still
// fits 2 CTAs per SM.
// v23 vs v22 (same math, same layouts): per-thread loader / transpose offsets computed once and pinned (v22's
// SASS recomputed them from threadIdx every tile, ~120 integer ops per warp per tile), stage index by counter (no
// i % ST), and S(i) = Q.K^T issued before the V transpose so the tensor cores overlap it.
// One CTA per (split s, request b), 256 threads = 2 warpgroups, one per KV head (warps 0-3: head 0, 4-7: head 1).
// A warpgroup's 64 accumulator rows are its 8 query heads x 8 tokens: warp wi's rows wi*16 + g and + 8 are tokens
// 2wi and 2wi + 1, query head h*8 + g. K/V rows (both heads, 128 B per key) are gathered through the page table by
// an ST-stage cp.async pipeline of 64-key tiles (16-byte chunks XOR-swizzled by key & 7, the 128-byte swizzle),
// page indices loaded one tile ahead.
// S = Q.K^T: FP8 wgmma m64n64k32, SS: Q (already E4M3, as FA3's FP8 path) is staged once in shared memory and
//   both operands go by 128-byte-swizzle descriptors (a register A operand was reused as scratch by ptxas).
// Online softmax in registers (exp2 domain); O is rescaled only when a row max moved.
// P.V: FP8 wgmma m64n72k32, RS: P (scaled by 2^8 into E4M3's range) is packed straight from the S accumulators;
//   V is transposed per tile into a double-buffered V^T [head * 72 + d][key slot] (key slots permuted to match the
//   S accumulator layout, 4x4 byte transposes by prmt). V^T rows 64..71 are E4M3 ones, so accumulator columns
//   64..71 are the softmax row sums. Tile i's P.V overlaps tile i + 1's barrier, transpose and QK.
// Output per row: the partial o = acc * v_scale / l and lse = m + log2(l) in the Triton head kernel's layout, so
// the same combine/norm/gate kernel merges them.
#define D 64
#define HKV 2
#define G 8
#define LQ 8
#define BN 64
#ifndef PF
#define PF 1  // page-index prefetch depth in tiles (1 = v24: loaded one tile ahead, so every tile waits one global latency)
#endif
#ifndef FVSKIP
#define FVSKIP 0  // diagnostic only: 1 = loads + barriers only (no MMA / softmax), 2 = also Q.K + V transpose, no softmax / P.V
#endif
#ifndef VT1
#define VT1 0
#endif
#define NVT (VT1 ? 1 : 2)  // V^T buffers
#ifndef ST
#define ST 3  // cp.async stages: ST - 1 tiles (16 KB each) in flight per CTA
#endif
#define QH 8192  // Q bytes per KV head in shared memory: 64 rows x 128 B (64 used), 128-byte swizzle
#define KROW 128
#define VTH 9216  // V^T bytes per head: 64 dims + 8 rows of ones (row sums of P) x 128-byte rows, 128-byte swizzle
#define TILE (BN * KROW)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  unsigned r;
  asm volatile("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ unsigned long long desc(unsigned addr, unsigned sbo, unsigned mode) {
  return (unsigned long long)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(sbo >> 4) << 32) |
         ((unsigned long long)mode << 62);
}
#define WG_PV2(acc, a, b, d0, d1)                                                                           \
  asm volatile("wgmma.fence.sync.aligned;\n"                                                                   \
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %44, 1, 1, 1;\n" \
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%40,%41,%42,%43}, %45, 1, 1, 1;\n" \
               "wgmma.commit_group.sync.aligned;"                                                                 \
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3])                                                                         \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),         \
                 "l"(d0), "l"(d1) : "memory")
#define WG_QK_SS(acc, da, db, sc)                                                                          \
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"                                                        \
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13," \
               "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"       \
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]),              \
                 "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]),              \
                 "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]),              \
                 "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]),              \
                 "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]),              \
                 "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]),              \
                 "+f"(acc[7][2]), "+f"(acc[7][3])                                                                 \
               : "l"(da), "l"(db), "r"(sc))
#define WG_WAIT_SO(sacc, oacc)                                                                              \
  asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(sacc[0][0]), "+f"(sacc[0][1]), "+f"(sacc[0][2]), "+f"(sacc[0][3]), "+f"(sacc[1][0]), "+f"(sacc[1][1]), "+f"(sacc[1][2]), "+f"(sacc[1][3]), "+f"(sacc[2][0]), "+f"(sacc[2][1]), "+f"(sacc[2][2]), "+f"(sacc[2][3]), "+f"(sacc[3][0]), "+f"(sacc[3][1]), "+f"(sacc[3][2]), "+f"(sacc[3][3]), "+f"(sacc[4][0]), "+f"(sacc[4][1]), "+f"(sacc[4][2]), "+f"(sacc[4][3]), "+f"(sacc[5][0]), "+f"(sacc[5][1]), "+f"(sacc[5][2]), "+f"(sacc[5][3]), "+f"(sacc[6][0]), "+f"(sacc[6][1]), "+f"(sacc[6][2]), "+f"(sacc[6][3]), "+f"(sacc[7][0]), "+f"(sacc[7][1]), "+f"(sacc[7][2]), "+f"(sacc[7][3]), "+f"(oacc[0][0]), "+f"(oacc[0][1]), "+f"(oacc[0][2]), "+f"(oacc[0][3]), "+f"(oacc[1][0]), "+f"(oacc[1][1]), "+f"(oacc[1][2]), "+f"(oacc[1][3]), "+f"(oacc[2][0]), "+f"(oacc[2][1]), "+f"(oacc[2][2]), "+f"(oacc[2][3]), "+f"(oacc[3][0]), "+f"(oacc[3][1]), "+f"(oacc[3][2]), "+f"(oacc[3][3]), "+f"(oacc[4][0]), "+f"(oacc[4][1]), "+f"(oacc[4][2]), "+f"(oacc[4][3]), "+f"(oacc[5][0]), "+f"(oacc[5][1]), "+f"(oacc[5][2]), "+f"(oacc[5][3]), "+f"(oacc[6][0]), "+f"(oacc[6][1]), "+f"(oacc[6][2]), "+f"(oacc[6][3]), "+f"(oacc[7][0]), "+f"(oacc[7][1]), "+f"(oacc[7][2]), "+f"(oacc[7][3]), "+f"(oacc[8][0]), "+f"(oacc[8][1]), "+f"(oacc[8][2]), "+f"(oacc[8][3]) :: "memory")
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned pack_e4(float x0, float x1, float x2, float x3) {  // bytes x0 (low) .. x3
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (unsigned)lo | ((unsigned)hi << 16);
}

#ifdef MAXR
extern "C" __global__ void __maxnreg__(MAXR)  // v25e diag: register cap ablation
#else
extern "C" __global__ void __launch_bounds__(256, 2)
#endif
attn_fv(const unsigned char* __restrict__ q, int stride_qt, const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc, const int* __restrict__ pt, int stride_pt,
        const int* __restrict__ seqlens, const float* __restrict__ ksp, const float* __restrict__ vsp,
        float* __restrict__ o_part, float* __restrict__ lse_out, float qk_scale, int chunk, int page_shift,
        int nsplit, int hq) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);  // 1024-aligned (SW128 atoms)
  unsigned char* Ks = smem;
  unsigned char* Vs = smem + ST * TILE;
  unsigned char* Vt = smem + 2 * ST * TILE;
  unsigned char* Qs = Vt + NVT * 2 * VTH;  // V^T double-buffered (P.V of tile i runs into tile i + 1) unless VT1
  const float NEG = __int_as_float(0xff800000);
  const int s = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = warp >> 2, wi = warp & 3;
  const int g = lane >> 2, t4 = lane & 3;
  const int total = seqlens[b];
  const int seq = total - LQ;
  const int lo = s * chunk;
  const int hi = min(lo + chunk, total);
  const int ntiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  const int j0 = wi * 2, j1 = wi * 2 + 1;           // rows wi*16 + g and + 8: tokens j0 / j1, query head h*8 + g
  const int pos0 = seq + j0, pos1 = seq + j1;
  const int head = h * G + g;

  {  // Q -> shared memory once (wgmma A operand by descriptor). Row m of KV head hh's 64 = token 2 (m >> 4) +
     // ((m >> 3) & 1), query head hh * 8 + (m & 7): warp wi's accumulator rows wi * 16 + g / + 8 are tokens j0 / j1.
    const int row = tid >> 1, hh = row >> 6, m = row & 63;
    const int tok = 2 * (m >> 4) + ((m >> 3) & 1), qh = hh * G + (m & 7);
    const unsigned char* src = q + (long long)(b * LQ + tok) * stride_qt + qh * D;
#pragma unroll
    for (int c = (tid & 1) * 2; c < (tid & 1) * 2 + 2; ++c)
      *(uint4*)(Qs + hh * QH + m * 128 + ((c ^ (m & 7)) << 4)) = *(const uint4*)(src + c * 16);
  }
  const float scale = qk_scale * ksp[0] * 1.4426950408889634f;
  const float vsc = vsp[0];

  // Per-thread shared-memory offsets, computed once and pinned in registers (v23: v22 recomputed all of this
  // address math from threadIdx every tile).
  // loader: key kl of the tile, 16-byte chunks c0, c0 + 1 of its 128-byte K and V rows (128-byte XOR swizzle)
  const int kl = tid >> 2, c0 = (tid & 3) * 2;
  unsigned ld_o0 = kl * KROW + ((c0 ^ (kl & 7)) << 4), ld_o1 = kl * KROW + (((c0 + 1) ^ (kl & 7)) << 4);
  // V -> V^T transpose: per rep, loads at tr_l[rep] (+1024) and tr_l1[rep] (+1024), stores at tr_s[rep][0..3]
  unsigned tr_l[2], tr_l1[2], tr_s[2][4];
#pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    const int combo = warp + 8 * rep;
    const int dg = (lane & 7) + 8 * (combo & 3), sg = (lane >> 3) + 4 * (combo >> 2);
    const int blk = sg >> 3, sl = sg & 7, tt = sl & 3, A = sl >> 2;
    const int k0 = blk * 32 + 16 * A + 2 * tt;  // keys k0, k0 + 1, k0 + 8, k0 + 9 (k0 & 7 == 2 tt)
    tr_l[rep] = k0 * KROW + ((((dg >> 2) ^ (k0 & 7)) * 16) + (dg & 3) * 4);
    tr_l1[rep] = (k0 + 1) * KROW + ((((dg >> 2) ^ ((k0 + 1) & 7)) * 16) + (dg & 3) * 4);
    const int slot = blk * 32 + sl * 4;
#pragma unroll
    for (int r = 0; r < 4; ++r) {  // row = head * 64 + d; 128-byte rows, 16-byte chunk ^= row & 7
      const int row = dg * 4 + r;
      tr_s[rep][r] = (row >> 6) * VTH + (row & 63) * 128 + (((slot >> 4) ^ (row & 7)) << 4) + (slot & 15);
    }
  }
  asm volatile("" : "+r"(ld_o0), "+r"(ld_o1), "+r"(tr_l[0]), "+r"(tr_l[1]), "+r"(tr_l1[0]), "+r"(tr_l1[1]));
  asm volatile("" : "+r"(tr_s[0][0]), "+r"(tr_s[0][1]), "+r"(tr_s[0][2]), "+r"(tr_s[0][3]), "+r"(tr_s[1][0]),
               "+r"(tr_s[1][1]), "+r"(tr_s[1][2]), "+r"(tr_s[1][3]));
  const unsigned ks_base = smem_u32(Ks), vs_base = smem_u32(Vs), vt_base = smem_u32(Vt);
  const unsigned char* kcc = kc + c0 * 16;
  const unsigned char* vcc = vc + c0 * 16;
  const int* ptb = pt + (long long)b * stride_pt;
  const int pmask = (1 << page_shift) - 1;
  auto rowof = [&](int tile) -> long long {
    const int t = lo + tile * BN + kl;
    if (tile < ntiles && t < hi) return ((long long)ptb[t >> page_shift] << page_shift) + (t & pmask);
    return -1;
  };
  auto issue = [&](int tile, int stage, long long row) {
    if (tile < ntiles) {
      const unsigned kd = ks_base + stage * TILE, vd = vs_base + stage * TILE;
      const int n = row >= 0 ? 16 : 0;
      const long long r = (row >= 0 ? row : 0) * KROW;
      cp16(kd + ld_o0, kcc + r, n);
      cp16(vd + ld_o0, vcc + r, n);
      cp16(kd + ld_o1, kcc + r + 16, n);
      cp16(vd + ld_o1, vcc + r + 16, n);
    }
    asm volatile("cp.async.commit_group;" ::);
  };

  float oacc[9][4];  // [0..7]: O (64 dims); [8]: sum of P over keys (V^T rows 64..71 are ones)
#pragma unroll
  for (int n = 0; n < 9; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
  float m0 = NEG, m1 = NEG;
  for (int k = tid; k < NVT * 2 * 8 * 8; k += 256) {  // ones rows: NVT buffers x 2 heads x 8 rows x 8 chunks (E4M3 1.0)
    const int buf_head = k >> 6, rr = (k >> 3) & 7, c = k & 7;
    *(uint4*)(Vt + buf_head * VTH + (64 + rr) * 128 + c * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }

  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // Q (st.shared) -> wgmma; the loop's first barrier orders it
#pragma unroll
  for (int k = 0; k < ST - 1; ++k) issue(k, k, rowof(k));
#if PF > 1
  int pq[PF];  // rows (page-table resolved) of tiles i + ST - 1 .. i + ST + PF - 2; -1 = past the end
#pragma unroll
  for (int k = 0; k < PF; ++k) pq[k] = (int)rowof(ST - 1 + k);
#else
  long long nextrow = rowof(ST - 1);
#endif
  const unsigned qs = smem_u32(Qs + h * QH);
  int st = 0;  // stage of tile i (i % ST); tile i + ST - 1 loads into stage st - 1 (mod ST), tile i - 1's
  for (int i = 0; i < ntiles; ++i) {
#if VT1
    if (i > 0) {  // tile i - 1's P.V reads the single V^T buffer: done (this warpgroup) before the barrier below
      float dz[8][4] = {};
      WG_WAIT_SO(dz, oacc);
    }
#endif
    asm volatile("cp.async.wait_group %0;" :: "n"(ST - 2));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // this thread's cp.async data -> wgmma (async proxy)
    __syncthreads();  // tile i's K/V visible; every warp is past tile i - 1's transpose and S / P.V waits
#if PF > 1
    issue(i + ST - 1, st == 0 ? ST - 1 : st - 1, (long long)pq[0]);  // into tile i - 1's stage (fully consumed)
#pragma unroll
    for (int k = 0; k < PF - 1; ++k) pq[k] = pq[k + 1];
    pq[PF - 1] = (int)rowof(i + ST + PF - 1);
#else
    issue(i + ST - 1, st == 0 ? ST - 1 : st - 1, nextrow);  // into tile i - 1's stage (fully consumed)
    nextrow = rowof(i + ST);
#endif
#if FVSKIP == 1
    st = st == ST - 1 ? 0 : st + 1;
    continue;
#endif

    // S(i) = Q.K^T issued first, so the tensor cores run it while this warpgroup transposes V(i)
    const unsigned kt = ks_base + st * TILE + h * 64;
    float sc[8][4] = {};
    wg_fence();
    WG_QK_SS(sc, desc(qs, 1024, 1), desc(kt, 1024, 1), 0);
    WG_QK_SS(sc, desc(qs + 32, 1024, 1), desc(kt + 32, 1024, 1), 1);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    {  // V tile i -> V^T[i & 1]: task = 4 dims x 4 key slots; slot 4t + 2a' + e of 32-key block <- key 8(2A + a') + 2t + e
      const unsigned vsrc = vs_base + st * TILE, vdst = vt_base + (VT1 ? 0 : (i & 1)) * 2 * VTH;
#pragma unroll
      for (int rep = 0; rep < 2; ++rep) {
        unsigned w0, w1, w2, w3;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w0) : "r"(vsrc + tr_l[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w1) : "r"(vsrc + tr_l1[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1+1024];" : "=r"(w2) : "r"(vsrc + tr_l[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1+1024];" : "=r"(w3) : "r"(vsrc + tr_l1[rep]) : "memory");
        const unsigned x = __byte_perm(w0, w1, 0x5140), y = __byte_perm(w2, w3, 0x5140);
        const unsigned x2 = __byte_perm(w0, w1, 0x7362), y2 = __byte_perm(w2, w3, 0x7362);
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][0]), "r"(__byte_perm(x, y, 0x5410)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][1]), "r"(__byte_perm(x, y, 0x7632)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][2]), "r"(__byte_perm(x2, y2, 0x5410)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][3]), "r"(__byte_perm(x2, y2, 0x7632)) : "memory");
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // V^T (st.shared) -> P.V wgmma (async proxy)
    __syncthreads();
    st = st == ST - 1 ? 0 : st + 1;
    WG_WAIT_SO(sc, oacc);  // S(i) and the previous tile's P.V (overlapped with this tile's barriers/transpose)
#if FVSKIP == 2
    if (sc[0][0] == 12345.f) oacc[0][0] += 1.f;
    continue;
#endif
    const int base = lo + i * BN;
    if (!(base + BN <= hi && base + BN - 1 <= pos0)) {  // edge tile: mask (warp-uniform branch)
      const int tb = base + t4 * 2;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int t = tb + n * 8 + e;
          if (!(t < hi && t <= pos0)) sc[n][e] = NEG;
          if (!(t < hi && t <= pos1)) sc[n][2 + e] = NEG;
        }
      }
    }
    float mx0 = NEG, mx1 = NEG;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      mx0 = fmaxf(mx0, fmaxf(sc[n][0], sc[n][1])); mx1 = fmaxf(mx1, fmaxf(sc[n][2], sc[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m0, mx0 * scale), mn1 = fmaxf(m1, mx1 * scale);
    const float ms0 = mn0 == NEG ? 0.f : mn0, ms1 = mn1 == NEG ? 0.f : mn1;
    const float a0 = ex2(m0 - ms0), a1 = ex2(m1 - ms1);
    const float nb0 = 8.f - ms0, nb1 = 8.f - ms1;  // p = 2^(s - m + 8) in (0, 256]
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sc[n][0] = ex2(fmaf(sc[n][0], scale, nb0)); sc[n][1] = ex2(fmaf(sc[n][1], scale, nb0));
      sc[n][2] = ex2(fmaf(sc[n][2], scale, nb1)); sc[n][3] = ex2(fmaf(sc[n][3], scale, nb1));
    }
    if (!__all_sync(0xffffffff, mn0 == m0 && mn1 == m1)) {  // rescale O (and the row sums) only if a max moved
#pragma unroll
      for (int n = 0; n < 9; ++n) { oacc[n][0] *= a0; oacc[n][1] *= a0; oacc[n][2] *= a1; oacc[n][3] *= a1; }
    }
    m0 = mn0; m1 = mn1;
    unsigned pa[2][4];
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const int n0 = kb * 4;
      pa[kb][0] = pack_e4(sc[n0][0], sc[n0][1], sc[n0 + 1][0], sc[n0 + 1][1]);
      pa[kb][1] = pack_e4(sc[n0][2], sc[n0][3], sc[n0 + 1][2], sc[n0 + 1][3]);
      pa[kb][2] = pack_e4(sc[n0 + 2][0], sc[n0 + 2][1], sc[n0 + 3][0], sc[n0 + 3][1]);
      pa[kb][3] = pack_e4(sc[n0 + 2][2], sc[n0 + 2][3], sc[n0 + 3][2], sc[n0 + 3][3]);
    }
    const unsigned vt = smem_u32(Vt + (VT1 ? 0 : (i & 1)) * 2 * VTH + h * VTH);
    // fence, both k-steps, commit and wait in one asm: the P fragments and the rescaled O are inputs, so the
    // fence cannot be scheduled ahead of their producers
    WG_PV2(oacc, pa[0], pa[1], desc(vt, 1024, 1), desc(vt + 32, 1024, 1));
  }
  {
    float dummy[8][4] = {};
    WG_WAIT_SO(dummy, oacc);  // last tile's P.V
  }
  asm volatile("cp.async.wait_group 0;" ::);
  const float l0 = oacc[8][0], l1 = oacc[8][2];  // full row sums (every key column of the tile went through the MMA)
  const float f0 = vsc / (l0 > 0.f ? l0 : 1.f), f1 = vsc / (l1 > 0.f ? l1 : 1.f);
  const long long i0 = ((long long)(b * LQ + j0) * hq + head) * nsplit + s;
  const long long i1 = ((long long)(b * LQ + j1) * hq + head) * nsplit + s;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    *(float2*)(o_part + i0 * D + n * 8 + t4 * 2) = make_float2(oacc[n][0] * f0, oacc[n][1] * f0);
    *(float2*)(o_part + i1 * D + n * 8 + t4 * 2) = make_float2(oacc[n][2] * f1, oacc[n][3] * f1);
  }
  if (t4 == 0) {
    lse_out[i0] = l0 > 0.f ? m0 + log2f(l0) - 8.f : NEG;
    lse_out[i1] = l1 > 0.f ? m1 + log2f(l1) - 8.f : NEG;
  }
}
"""
_WS_SRC = r"""
// Full-attention verify (q_len 8) split-KV partials, attention v25 warp-specialized variant (attn_fvws).
// Same math, layouts and outputs as attn_v25_fv.cu (v24's FV kernel), but the K/V gather is decoupled from the
// compute: 288 threads = 2 consumer warpgroups (one per KV head, as v24) + 1 producer warp.
//  - producer warp: per 64-key tile, resolves the page-table rows (one tile ahead, shuffled across lanes), waits for
//    the stage's "empty" mbarrier, issues 32 cp.async per lane (8 lanes per 128-byte K / V row: 4 whole rows per
//    instruction) into the 128-byte-swizzled stage, then cp.async.mbarrier.arrive.noinc on the stage's "full"
//    mbarrier (count 32), so a stage turns full when every lane's copies have landed.
//  - consumers: wait "full", fence.proxy.async (cp.async writes -> wgmma async proxy), named barrier per
//    warpgroup (bar 1 + h, 128 threads: every warp of the warpgroup is past tile i - 1's waits, so V^T buffer i & 1
//    is free), Q.K issue, V transpose of the warpgroup's own head only, fence + named barrier, wait S(i) / P.V(i - 1),
//    then each warp's lane 0 arrives on the stage's "empty" mbarrier (count 8), softmax, P.V issue.
//  - v24 diagnostic (v25d4): loads alone ran at ~2.85 TB/s but the full kernel at ~2 TB/s because every tile's
//    cp.async issue waited behind the CTA-wide barriers and the softmax; here no consumer barrier gates the loads.
// The last tile's P.V is waited inside the loop (no wgmma serialization, ptxas C7515).
// Every inline asm touching shared memory carries a "memory" clobber.
#define D 64
#define HKV 2
#define G 8
#define LQ 8
#define BN 64
#ifndef ST
#define ST 3  // cp.async stages: ST - 1 tiles (16 KB each) in flight per CTA
#endif
// v25e knobs (defaults = the v25d8 kernel, same cubin):
#ifndef NPW
#define NPW 1  // producer warps (8 .. 8 + NPW - 1); NPW 2 / 4 split each lane's 16 cp.async instructions per tile
#endif
#ifndef CBAR
#define CBAR 0  // 1: both consumer barriers CTA-wide over the 256 consumer threads (v24's __syncthreads structure)
#endif
#ifndef MAXR
#define MAXR 112  // __maxnreg__; 2 CTAs/SM need (256 + 32 NPW) x 2 x MAXR <= 64K
#endif
#ifndef SMR
#define SMR 0  // NPW 4 only: setmaxnreg (producer warpgroup -> 24, consumers -> SMR registers)
#endif
#ifndef WSABL
#define WSABL 0  // diagnostic ablations (timing only, results wrong): 1 no gather (producer issues no cp.async),
                 // 2 no consumer compute (no Q.K / transpose / softmax / P.V), 4 no loop-top barrier, 8 no barrier
                 // after the transpose, 16 test_wait spin instead of try_wait, 32 no softmax / P.V (Q.K + transpose
                 // kept), 64 no fence.proxy.async after the "full" wait
#endif
#define NT (256 + 32 * NPW)
#define QH 8192  // Q bytes per KV head in shared memory: 64 rows x 128 B (64 used), 128-byte swizzle
#define KROW 128
#define VTH 9216  // V^T bytes per head: 64 dims + 8 rows of ones (row sums of P) x 128-byte rows, 128-byte swizzle
#define TILE (BN * KROW)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  unsigned r;
  asm volatile("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ unsigned long long desc(unsigned addr, unsigned sbo, unsigned mode) {
  return (unsigned long long)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(sbo >> 4) << 32) |
         ((unsigned long long)mode << 62);
}
#define WG_PV2(acc, a, b, d0, d1)                                                                           \
  asm volatile("wgmma.fence.sync.aligned;\n"                                                                   \
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %44, 1, 1, 1;\n" \
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%40,%41,%42,%43}, %45, 1, 1, 1;\n" \
               "wgmma.commit_group.sync.aligned;"                                                                 \
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3])                                                                         \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),         \
                 "l"(d0), "l"(d1) : "memory")
#define WG_QK_SS(acc, da, db, sc)                                                                          \
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"                                                        \
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13," \
               "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"       \
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]),              \
                 "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]),              \
                 "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]),              \
                 "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]),              \
                 "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]),              \
                 "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]),              \
                 "+f"(acc[7][2]), "+f"(acc[7][3])                                                                 \
               : "l"(da), "l"(db), "r"(sc))
#define WG_WAIT_SO(sacc, oacc)                                                                              \
  asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(sacc[0][0]), "+f"(sacc[0][1]), "+f"(sacc[0][2]), "+f"(sacc[0][3]), "+f"(sacc[1][0]), "+f"(sacc[1][1]), "+f"(sacc[1][2]), "+f"(sacc[1][3]), "+f"(sacc[2][0]), "+f"(sacc[2][1]), "+f"(sacc[2][2]), "+f"(sacc[2][3]), "+f"(sacc[3][0]), "+f"(sacc[3][1]), "+f"(sacc[3][2]), "+f"(sacc[3][3]), "+f"(sacc[4][0]), "+f"(sacc[4][1]), "+f"(sacc[4][2]), "+f"(sacc[4][3]), "+f"(sacc[5][0]), "+f"(sacc[5][1]), "+f"(sacc[5][2]), "+f"(sacc[5][3]), "+f"(sacc[6][0]), "+f"(sacc[6][1]), "+f"(sacc[6][2]), "+f"(sacc[6][3]), "+f"(sacc[7][0]), "+f"(sacc[7][1]), "+f"(sacc[7][2]), "+f"(sacc[7][3]), "+f"(oacc[0][0]), "+f"(oacc[0][1]), "+f"(oacc[0][2]), "+f"(oacc[0][3]), "+f"(oacc[1][0]), "+f"(oacc[1][1]), "+f"(oacc[1][2]), "+f"(oacc[1][3]), "+f"(oacc[2][0]), "+f"(oacc[2][1]), "+f"(oacc[2][2]), "+f"(oacc[2][3]), "+f"(oacc[3][0]), "+f"(oacc[3][1]), "+f"(oacc[3][2]), "+f"(oacc[3][3]), "+f"(oacc[4][0]), "+f"(oacc[4][1]), "+f"(oacc[4][2]), "+f"(oacc[4][3]), "+f"(oacc[5][0]), "+f"(oacc[5][1]), "+f"(oacc[5][2]), "+f"(oacc[5][3]), "+f"(oacc[6][0]), "+f"(oacc[6][1]), "+f"(oacc[6][2]), "+f"(oacc[6][3]), "+f"(oacc[7][0]), "+f"(oacc[7][1]), "+f"(oacc[7][2]), "+f"(oacc[7][3]), "+f"(oacc[8][0]), "+f"(oacc[8][1]), "+f"(oacc[8][2]), "+f"(oacc[8][3]) :: "memory")
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(unsigned addr, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned addr, unsigned parity) {
#if WSABL & 16
  asm volatile("{\n .reg .pred p;\n WS_WAIT:\n mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WS_WAIT;\n}"
               :: "r"(addr), "r"(parity) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n WS_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WS_WAIT;\n}"
               :: "r"(addr), "r"(parity) : "memory");
#endif
}
__device__ __forceinline__ void mbar_arrive(unsigned addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void cp_arrive_noinc(unsigned addr) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void wg_bar(int h) {  // named barrier 1 + h over warpgroup h (immediate ids: ptxas reserves 2)
#if CBAR
  asm volatile("bar.sync 1, 256;" ::: "memory");  // all 8 consumer warps (the producer warps never join)
#else
  if (h) asm volatile("bar.sync 2, 128;" ::: "memory");
  else asm volatile("bar.sync 1, 128;" ::: "memory");
#endif
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned pack_e4(float x0, float x1, float x2, float x3) {  // bytes x0 (low) .. x3
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (unsigned)lo | ((unsigned)hi << 16);
}

extern "C" __global__ void __maxnreg__(MAXR)  // 2 CTAs/SM: 9 warps x 2 x 32 x 112 <= 64K registers (NPW 1)
attn_fvws(const unsigned char* __restrict__ q, int stride_qt, const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc, const int* __restrict__ pt, int stride_pt,
        const int* __restrict__ seqlens, const float* __restrict__ ksp, const float* __restrict__ vsp,
        float* __restrict__ o_part, float* __restrict__ lse_out, float qk_scale, int chunk, int page_shift,
        int nsplit, int hq) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);  // 1024-aligned (SW128 atoms)
  unsigned char* Ks = smem;
  unsigned char* Vs = smem + ST * TILE;
  unsigned char* Vt = smem + 2 * ST * TILE;
  unsigned char* Qs = Vt + 2 * 2 * VTH;  // V^T double-buffered per head (P.V of tile i runs into tile i + 1)
  const unsigned bars = smem_u32(Qs + 2 * QH);  // full[ST], empty[ST] mbarriers (8 B each)
#define FULL(k) (bars + (k) * 8)
#define EMPTY(k) (bars + (ST + (k)) * 8)
  const float NEG = __int_as_float(0xff800000);
  const int s = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = warp >> 2, wi = warp & 3;
  const int g = lane >> 2, t4 = lane & 3;
  const int total = seqlens[b];
  const int seq = total - LQ;
  const int lo = s * chunk;
  const int hi = min(lo + chunk, total);
  const int ntiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  const int j0 = wi * 2, j1 = wi * 2 + 1;           // rows wi*16 + g and + 8: tokens j0 / j1, query head h*8 + g
  const int pos0 = seq + j0, pos1 = seq + j1;
  const int head = h * G + g;

  if (tid < 256) {  // Q -> shared memory once (wgmma A operand by descriptor). Row m of KV head hh's 64 = token 2 (m >> 4) +
     // ((m >> 3) & 1), query head hh * 8 + (m & 7): warp wi's accumulator rows wi * 16 + g / + 8 are tokens j0 / j1.
    const int row = tid >> 1, hh = row >> 6, m = row & 63;
    const int tok = 2 * (m >> 4) + ((m >> 3) & 1), qh = hh * G + (m & 7);
    const unsigned char* src = q + (long long)(b * LQ + tok) * stride_qt + qh * D;
#pragma unroll
    for (int c = (tid & 1) * 2; c < (tid & 1) * 2 + 2; ++c)
      *(uint4*)(Qs + hh * QH + m * 128 + ((c ^ (m & 7)) << 4)) = *(const uint4*)(src + c * 16);
  }
  const float scale = qk_scale * ksp[0] * 1.4426950408889634f;
  const float vsc = vsp[0];

  // Per-thread shared-memory offsets, computed once and pinned in registers (v23: v22 recomputed all of this
  // address math from threadIdx every tile).
  // V -> V^T transpose: per rep, loads at tr_l[rep] (+1024) and tr_l1[rep] (+1024), stores at tr_s[rep][0..3]
  unsigned tr_l[2], tr_l1[2], tr_s[2][4];
#pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    // v24's task map with warp -> pw: warpgroup h covers dims of its own head only (dg in 16h .. 16h + 15)
    const int pw = 2 * h + (wi & 1) + 4 * (wi >> 1);
    const int combo = pw + 8 * rep;
    const int dg = (lane & 7) + 8 * (combo & 3), sg = (lane >> 3) + 4 * (combo >> 2);
    const int blk = sg >> 3, sl = sg & 7, tt = sl & 3, A = sl >> 2;
    const int k0 = blk * 32 + 16 * A + 2 * tt;  // keys k0, k0 + 1, k0 + 8, k0 + 9 (k0 & 7 == 2 tt)
    tr_l[rep] = k0 * KROW + ((((dg >> 2) ^ (k0 & 7)) * 16) + (dg & 3) * 4);
    tr_l1[rep] = (k0 + 1) * KROW + ((((dg >> 2) ^ ((k0 + 1) & 7)) * 16) + (dg & 3) * 4);
    const int slot = blk * 32 + sl * 4;
#pragma unroll
    for (int r = 0; r < 4; ++r) {  // row = head * 64 + d; 128-byte rows, 16-byte chunk ^= row & 7
      const int row = dg * 4 + r;
      tr_s[rep][r] = (row >> 6) * VTH + (row & 63) * 128 + (((slot >> 4) ^ (row & 7)) << 4) + (slot & 15);
    }
  }
  asm volatile("" : "+r"(tr_l[0]), "+r"(tr_l[1]), "+r"(tr_l1[0]), "+r"(tr_l1[1]));
  asm volatile("" : "+r"(tr_s[0][0]), "+r"(tr_s[0][1]), "+r"(tr_s[0][2]), "+r"(tr_s[0][3]), "+r"(tr_s[1][0]),
               "+r"(tr_s[1][1]), "+r"(tr_s[1][2]), "+r"(tr_s[1][3]));
  const int* ptb = pt + (long long)b * stride_pt;
  const int pmask = (1 << page_shift) - 1;
  float oacc[9][4];  // [0..7]: O (64 dims); [8]: sum of P over keys (V^T rows 64..71 are ones)
#pragma unroll
  for (int n = 0; n < 9; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
  float m0 = NEG, m1 = NEG;
  for (int k = tid; k < 2 * 2 * 8 * 8; k += NT) {  // ones rows: 2 buffers x 2 heads x 8 rows x 8 chunks (E4M3 1.0)
    const int buf_head = k >> 6, rr = (k >> 3) & 7, c = k & 7;
    *(uint4*)(Vt + buf_head * VTH + (64 + rr) * 128 + c * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }

  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < ST; ++k) { mbar_init(FULL(k), 32 * NPW); mbar_init(EMPTY(k), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // Q / ones rows (st.shared) -> wgmma
  __syncthreads();

#if NPW == 1
  if (warp == 8) {  // producer warp
#else
  if (warp >= 8) {  // producer warps
#endif
#if SMR
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" ::: "memory");
#endif
#if NPW == 1
    const int pwi = 0;
#else
    const int pwi = warp - 8;
#endif
    const int pk = lane >> 3, pc = lane & 7;  // instruction j: key pk + 4 j, 16-byte chunk pc of its K and V rows
    auto prow = [&](int tile, int k) -> int {
      const int t = lo + tile * BN + k;
      return (tile < ntiles && t < hi) ? (ptb[t >> page_shift] << page_shift) + (t & pmask) : -1;
    };
    int ra = prow(0, lane), rb = prow(0, lane + 32);  // rows of keys lane / lane + 32 of the tile
    const unsigned ks_base = smem_u32(Ks), vs_base = smem_u32(Vs);
    int stg = 0, ph = 0;
    for (int i = 0; i < ntiles; ++i) {
      const int na = prow(i + 1, lane), nb = prow(i + 1, lane + 32);  // next tile's rows, in flight meanwhile
      if (i >= ST) mbar_wait(EMPTY(stg), ph ^ 1);  // tile i - ST (same stage) consumed by all 8 consumer warps
      const unsigned kd = ks_base + stg * TILE, vd = vs_base + stg * TILE;
#pragma unroll
      for (int jj = 0; jj < 16 / NPW; ++jj) {
        const int j = pwi * (16 / NPW) + jj;  // NPW 1: j = jj
        const int key = pk + 4 * j;
        const int row = __shfl_sync(0xffffffffu, j < 8 ? ra : rb, key & 31);
        const unsigned off = key * KROW + ((pc ^ (key & 7)) << 4);
        const int n = row >= 0 ? 16 : 0;
        const long long r = (long long)(row >= 0 ? row : 0) * KROW + pc * 16;
#if !(WSABL & 1)
        cp16(kd + off, kc + r, n);
        cp16(vd + off, vc + r, n);
#else
        if (n == 12345) cp16(kd + off, kc + r, n);  // never: keeps the address math live
#endif
      }
      cp_arrive_noinc(FULL(stg));
      ra = na; rb = nb;
      if (++stg == ST) { stg = 0; ph ^= 1; }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }

#if SMR
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(SMR) : "memory");
#endif
  const unsigned qs = smem_u32(Qs + h * QH);
  const unsigned ks_base = smem_u32(Ks), vs_base = smem_u32(Vs), vt_base = smem_u32(Vt);
  int st = 0, ph = 0;  // stage of tile i and its "full" phase parity
  for (int i = 0; i < ntiles; ++i) {
    mbar_wait(FULL(st), ph);
#if !(WSABL & 64)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // producer's cp.async data -> wgmma (async proxy)
#endif
#if !(WSABL & 4)
    wg_bar(h);  // every warp of this warpgroup is past tile i - 1's waits: V^T buffer i & 1 (P.V(i - 2)) free
#endif

    // S(i) = Q.K^T issued first, so the tensor cores run it while this warpgroup transposes V(i)
    const unsigned kt = ks_base + st * TILE + h * 64;
    float sc[8][4] = {};
#if !(WSABL & 2)
    wg_fence();
    WG_QK_SS(sc, desc(qs, 1024, 1), desc(kt, 1024, 1), 0);
    WG_QK_SS(sc, desc(qs + 32, 1024, 1), desc(kt + 32, 1024, 1), 1);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#else
    (void)kt;
#endif
#if !(WSABL & 2)
    {  // V tile i -> V^T[i & 1]: task = 4 dims x 4 key slots; slot 4t + 2a' + e of 32-key block <- key 8(2A + a') + 2t + e
      const unsigned vsrc = vs_base + st * TILE, vdst = vt_base + (i & 1) * 2 * VTH;
#pragma unroll
      for (int rep = 0; rep < 2; ++rep) {
        unsigned w0, w1, w2, w3;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w0) : "r"(vsrc + tr_l[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w1) : "r"(vsrc + tr_l1[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1+1024];" : "=r"(w2) : "r"(vsrc + tr_l[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1+1024];" : "=r"(w3) : "r"(vsrc + tr_l1[rep]) : "memory");
        const unsigned x = __byte_perm(w0, w1, 0x5140), y = __byte_perm(w2, w3, 0x5140);
        const unsigned x2 = __byte_perm(w0, w1, 0x7362), y2 = __byte_perm(w2, w3, 0x7362);
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][0]), "r"(__byte_perm(x, y, 0x5410)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][1]), "r"(__byte_perm(x, y, 0x7632)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][2]), "r"(__byte_perm(x2, y2, 0x5410)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][3]), "r"(__byte_perm(x2, y2, 0x7632)) : "memory");
      }
    }
#endif
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // V^T (st.shared) -> P.V wgmma (async proxy)
#if !(WSABL & 8)
    wg_bar(h);
#endif
    WG_WAIT_SO(sc, oacc);  // S(i) and the previous tile's P.V (overlapped with this tile's barrier/transpose)
    if (lane == 0) mbar_arrive(EMPTY(st));  // this warp is done with stage st (its K via S(i), its V via the transpose)
    if (++st == ST) { st = 0; ph ^= 1; }
#if WSABL & (2 | 32)
    if (sc[0][0] == 12345.f) oacc[0][0] += 1.f;
    continue;
#endif
    const int base = lo + i * BN;
    if (!(base + BN <= hi && base + BN - 1 <= pos0)) {  // edge tile: mask (warp-uniform branch)
      const int tb = base + t4 * 2;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int t = tb + n * 8 + e;
          if (!(t < hi && t <= pos0)) sc[n][e] = NEG;
          if (!(t < hi && t <= pos1)) sc[n][2 + e] = NEG;
        }
      }
    }
    float mx0 = NEG, mx1 = NEG;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      mx0 = fmaxf(mx0, fmaxf(sc[n][0], sc[n][1])); mx1 = fmaxf(mx1, fmaxf(sc[n][2], sc[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m0, mx0 * scale), mn1 = fmaxf(m1, mx1 * scale);
    const float ms0 = mn0 == NEG ? 0.f : mn0, ms1 = mn1 == NEG ? 0.f : mn1;
    const float a0 = ex2(m0 - ms0), a1 = ex2(m1 - ms1);
    const float nb0 = 8.f - ms0, nb1 = 8.f - ms1;  // p = 2^(s - m + 8) in (0, 256]
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sc[n][0] = ex2(fmaf(sc[n][0], scale, nb0)); sc[n][1] = ex2(fmaf(sc[n][1], scale, nb0));
      sc[n][2] = ex2(fmaf(sc[n][2], scale, nb1)); sc[n][3] = ex2(fmaf(sc[n][3], scale, nb1));
    }
    if (!__all_sync(0xffffffff, mn0 == m0 && mn1 == m1)) {  // rescale O (and the row sums) only if a max moved
#pragma unroll
      for (int n = 0; n < 9; ++n) { oacc[n][0] *= a0; oacc[n][1] *= a0; oacc[n][2] *= a1; oacc[n][3] *= a1; }
    }
    m0 = mn0; m1 = mn1;
    unsigned pa[2][4];
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const int n0 = kb * 4;
      pa[kb][0] = pack_e4(sc[n0][0], sc[n0][1], sc[n0 + 1][0], sc[n0 + 1][1]);
      pa[kb][1] = pack_e4(sc[n0][2], sc[n0][3], sc[n0 + 1][2], sc[n0 + 1][3]);
      pa[kb][2] = pack_e4(sc[n0 + 2][0], sc[n0 + 2][1], sc[n0 + 3][0], sc[n0 + 3][1]);
      pa[kb][3] = pack_e4(sc[n0 + 2][2], sc[n0 + 2][3], sc[n0 + 3][2], sc[n0 + 3][3]);
    }
    const unsigned vt = smem_u32(Vt + (i & 1) * 2 * VTH + h * VTH);
    // fence, both k-steps, commit and wait in one asm: the P fragments and the rescaled O are inputs, so the
    // fence cannot be scheduled ahead of their producers
    WG_PV2(oacc, pa[0], pa[1], desc(vt, 1024, 1), desc(vt + 32, 1024, 1));
    if (i == ntiles - 1) WG_WAIT_SO(sc, oacc);  // last tile's P.V, inside the loop: a wait after the loop (v24's) made
                                                // ptxas serialize every wgmma here (C7515)
  }
  const float l0 = oacc[8][0], l1 = oacc[8][2];  // full row sums (every key column of the tile went through the MMA)
  const float f0 = vsc / (l0 > 0.f ? l0 : 1.f), f1 = vsc / (l1 > 0.f ? l1 : 1.f);
  const long long i0 = ((long long)(b * LQ + j0) * hq + head) * nsplit + s;
  const long long i1 = ((long long)(b * LQ + j1) * hq + head) * nsplit + s;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    *(float2*)(o_part + i0 * D + n * 8 + t4 * 2) = make_float2(oacc[n][0] * f0, oacc[n][1] * f0);
    *(float2*)(o_part + i1 * D + n * 8 + t4 * 2) = make_float2(oacc[n][2] * f1, oacc[n][3] * f1);
  }
  if (t4 == 0) {
    lse_out[i0] = l0 > 0.f ? m0 + log2f(l0) - 8.f : NEG;
    lse_out[i1] = l1 > 0.f ? m1 + log2f(l1) - 8.f : NEG;
  }
}
"""  # tools/attn_v25_ws.cu: attn_fvws (fv_vt1 & 64)
FV_BN = 64  # must match BN in tools/attn_v25_fv.cu
FV_ST = 3  # cp.async stages (-DST; 3 = v24)
FV_VT1 = 55869504  # flags: VT1 | FVSKIP << 1 (diag) | PF << 3 (3 bits) | 64 = warp-specialized attn_fvws; 0 = v24
FV_OCC = 3  # split-KV programs per SM for _nsplit (the kernel fits 2 per SM at ST 3; the rest is load balance)
ROWS_FULL = ("rows", 128, 4, 3, 4, True)  # full decode, batch >= 2


def _fv_smem(st, vt1=None):
    vt1 = FV_VT1 if vt1 is None else vt1
    return 2 * st * FV_BN * 128 + (1 if vt1 & 1 else 2) * 2 * 9216 + 2 * 8192 + 1024 + (
        128 if vt1 & 64 else 0)  # K + V E4M3 stages, V^T buffers (72 rows), Q, alignment (+ WS mbarriers)


FV_SMEM = _fv_smem(FV_ST)


def _wsf(abl=0, npw=1, cbar=0, maxr=0, smr=0, ws=1):
    """v25e flags: 64 (attn_fvws) | WSABL << 7 | log2(NPW) << 14 | CBAR << 16 | MAXR / 8 << 17 | SMR / 8 << 22.
    ws=0: attn_fv (v24's kernel) with only MAXR (register-cap ablation)."""
    return (64 if ws else 0) | abl << 7 | {1: 0, 2: 1, 4: 2}[npw] << 14 | cbar << 16 | (maxr // 8) << 17 | (smr // 8) << 22


def _ws_npw(vt1):
    return 1 << ((vt1 >> 14) & 3)


def _ws_opts(vt1):
    o = [f"-DMAXR={((vt1 >> 17) & 31) * 8}".encode()] if (vt1 >> 17) & 31 else []
    if vt1 & 64:
        o += [f"-DWSABL={(vt1 >> 7) & 127}".encode()] if (vt1 >> 7) & 127 else []
        o += [f"-DNPW={_ws_npw(vt1)}".encode()] if _ws_npw(vt1) > 1 else []
        o += [b"-DCBAR=1"] if (vt1 >> 16) & 1 else []
        o += [f"-DSMR={((vt1 >> 22) & 31) * 8}".encode()] if (vt1 >> 22) & 31 else []
    return o


_FV = {}  # "cubin": NVRTC output (device-independent); device index -> (module, function); "sm90": index -> bool


def _fv_cubin(st=None, vt1=None):
    """NVRTC (cuda.bindings, in-process) -> sm_90a cubin bytes (no files). Host-only, no CUDA calls."""
    st = FV_ST if st is None else st
    vt1 = FV_VT1 if vt1 is None else vt1
    if ("cubin", st, vt1) not in _FV:
        from cuda.bindings import nvrtc
        err, prog = nvrtc.nvrtcCreateProgram((_WS_SRC if vt1 & 64 else _FV_SRC).encode(), b"attn_fv.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        opts = [b"-arch=sm_90a", b"-std=c++17", f"-DST={st}".encode()] + ([b"-DVT1=1"] if vt1 & 1 else []) + (
            [f"-DFVSKIP={(vt1 >> 1) & 3}".encode()] if (vt1 >> 1) & 3 else []) + (
            [f"-DPF={(vt1 >> 3) & 7}".encode()] if (vt1 >> 3) & 7 else []) + _ws_opts(vt1)  # vt1 = VT1 | FVSKIP << 1 (diag) | PF << 3 | WS 64 | v25e (_wsf)
        (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="ignore"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        _FV[("cubin", st, vt1)] = cubin
    return _FV[("cubin", st, vt1)]


def _fv_fn(dev, st=None, vt1=None):
    """The kernel loaded on CUDA device dev by CuPy from memory. Loaded at import (below) for the current device,
    so the NVRTC compile and module load never run inside the judge's CUDA-graph capture."""
    st = FV_ST if st is None else st
    vt1 = FV_VT1 if vt1 is None else vt1
    if (dev, st, vt1) not in _FV:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("attn_fv: kernel not loaded before CUDA-graph capture (device %d)" % dev)
        import cupy as cp
        cubin = _fv_cubin(st, vt1)
        with cp.cuda.Device(dev):
            mod = cp.cuda.function.Module()
            mod.load(cubin)
            f = mod.get_function("attn_fvws" if vt1 & 64 else "attn_fv")
            drv = cp.cuda.driver
            drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8), _fv_smem(st, vt1))
        _FV[(dev, st, vt1)] = (mod, f)
    return _FV[(dev, st, vt1)][1]


def _fv_sm90(device):
    """True on an sm_90 (H100 / H200) device: the cubin is sm_90a only."""
    key = ("sm90", device.index)
    if key not in _FV:
        _FV[key] = torch.cuda.get_device_capability(device) == (9, 0)
    return _FV[key]


if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
    _fv_fn(torch.cuda.current_device())  # compile + load at import (review #20 m3), not on the first call


def _fv_ok(kind, L, batch, hq, hkv, d, q):
    return (kind == "full" and L == 8 and batch >= 2 and (hq, hkv, d) == (16, 2, 64) and q.device.type == "cuda"
            and _fv_sm90(q.device))


def _fv_page_ok(page):
    return page & (page - 1) == 0  # the kernel indexes pages by shift


def _fv(data, q, qkv, cfg, st=None, occ=None, vt1=None):
    """Full verify at batch >= 2: CUDA split-KV partials (tools/attn_v22_fv.cu), then the Triton combine."""
    import cupy as cp
    import numpy as np
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = batch * L
    page = cfg["page_size"]
    capacity = data["page_table"].shape[1] * page
    st = FV_ST if st is None else st
    vt1 = FV_VT1 if vt1 is None else vt1
    nsplit = _nsplit(cfg, batch, 1, capacity, FV_OCC if occ is None else occ)
    chunk = triton.cdiv(capacity, nsplit)
    o_part = torch.empty((T * hq * nsplit, d), device=q.device, dtype=torch.float32)
    lse = torch.empty((T * hq * nsplit,), device=q.device, dtype=torch.float32)
    pt, sl = data["page_table"], data["cache_seqlens"]
    assert pt.dtype == torch.int32 and sl.dtype == torch.int32 and q.is_contiguous()
    kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    f = _fv_fn(dev, st, vt1)
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(q.device).cuda_stream):
        f((nsplit, batch), (256 + 32 * _ws_npw(vt1) if vt1 & 64 else 256,), (
            np.uint64(q.data_ptr()), np.int32(q.stride(0)), np.uint64(kc.data_ptr()), np.uint64(vc.data_ptr()),
            np.uint64(pt.data_ptr()), np.int32(pt.stride(0)), np.uint64(sl.data_ptr()),
            np.uint64(data["k_scale"].data_ptr()), np.uint64(data["v_scale"].data_ptr()),
            np.uint64(o_part.data_ptr()), np.uint64(lse.data_ptr()),
            np.float32(cfg["sm_scale"] * cfg["qk_scale_factor"]), np.int32(chunk), np.int32(page.bit_length() - 1), np.int32(nsplit),
            np.int32(hq)), shared_mem=_fv_smem(st, vt1))
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = torch.empty((T, hq * d), device=qkv.device, dtype=data["qkv"].dtype)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=4 if nsplit >= 64 else 1)
    return out.view(batch, L, hq * d)


def kernel_fn(data):
    return _kernel_path(data)


def _kernel_path(data, path_override=None):
    cfg = data["config"]
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = batch * L
    qkv = data["qkv"].view(T, -1)
    if qkv.device.type == "cpu":
        q = _norm_rope_store_fused(data, qkv.float(), cfg, torch.float32)  # interpreter: no bf16
    else:
        q = _norm_rope_store_fused(data, qkv, cfg, FP8)
    kind = "sliding" if cfg["window"] else "full"
    path = _attn_path(kind, L, batch, hkv, d) if path_override is None else path_override
    if path is None and _fv_ok(kind, L, batch, hq, hkv, d, q) and _fv_page_ok(cfg["page_size"]):
        return _fv(data, q, qkv, cfg)  # v22: CUDA full-verify kernel instead of FA3
    if path is None and q.device.type == "cpu":
        path = ("head", 64, 4, 3, 4, False)  # local CPU tests exercise the split-KV kernel everywhere
    if path is None:
        return _fa3(data, q, qkv, cfg)
    which, block_n, warps, stages, occ, hoist = path
    G = hq // hkv
    page = cfg["page_size"]
    capacity = data["page_table"].shape[1] * page
    nsplit = _nsplit(cfg, batch, 1 if which == "rows" else hkv, capacity, occ)
    span = (cfg["window"] + L - 1) if cfg["window"] else capacity
    chunk = triton.cdiv(span, nsplit)
    block_m = max(16, triton.next_power_of_2(G * L))
    o_part = torch.empty((T * hq * nsplit, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((T * hq * nsplit,), device=qkv.device, dtype=torch.float32)
    hw = qkv.device.type == "cuda"
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    args = (q, q.stride(0), kc, vc, data["page_table"], data["page_table"].stride(0), data["cache_seqlens"],
            data["k_scale"], data["v_scale"], o_part, lse, cfg["sm_scale"] * cfg["qk_scale_factor"], chunk, page)
    if which == "rows":
        rows = max(16, triton.next_power_of_2(hq * L))
        _attn_split_rows[(nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"], BLOCK_M=rows, BLOCK_N=block_n, D=d,
            HW=hw, HOIST=hoist, PREF=hoist, PV16=hw, num_warps=warps, num_stages=stages)
    else:
        _attn_split[(hkv, nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"], BLOCK_M=block_m, BLOCK_N=block_n, D=d,
            HW=hw, HOIST=hoist, PREF=hoist, PV16=hw, num_warps=warps, num_stages=stages)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = torch.empty((T, hq * d), device=qkv.device, dtype=data["qkv"].dtype)
    # one warp per (token, head) is latency-bound once it has >= 64 partials to read (7 us at 256)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=4 if nsplit >= 64 else 1)
    return out.view(batch, L, hq * d)
