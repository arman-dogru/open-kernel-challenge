# okc: problem=muse_attention_decode
"""Gated attention decode with a split-KV (flash-decoding) Triton kernel over the paged FP8
cache, and a combine kernel that also applies the per-head RMSNorm and sigmoid gate.

Kept from SGLang: FusedQKNorm (QK-norm + interleaved RoPE) and store_cache (FP8 cache write).
Replaced, on the shapes in ATTN_CFG: the query cast, FA3 and rms_norm_gate. Other scored shapes
keep SGLang's FA3 path (where it is still faster).

- Attention runs over every key it should (whole window / whole context, causal inside the
  verify block); no skipping. Queries stay BF16 (finer than FA3's E4M3), K/V are read as FP8
  E4M3 and decoded exactly in registers, softmax and accumulation in FP32.
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
_NORMS = {}
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
                D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False):
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
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
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
                     D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False):
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
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
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
    """Merge the splits of one (token, head), then RMSNorm over the head x sigmoid(gate)."""
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


def _qk_norm(hq, hkv, d):
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _norm_rope_store(data, qkv, cfg):
    """SGLang's FusedQKNorm + store_cache, as the baseline runs them. Returns q [T, Hq*64]."""
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = qkv.shape[0]
    if qkv.device.type == "cpu":  # local CPU testing only: a torch stand-in
        return _norm_rope_store_torch(data, qkv, cfg)
    from sglang.kernels.ops.kvcache.kvcache import store_cache
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, qkv.device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    k = k.reshape(tokens, hkv * d)
    if cfg["kv_scale"] != 1.0:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                data["k_cache"].view(-1, hkv * d).view(torch.uint8),
                data["v_cache"].view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    return q.reshape(tokens, hq * d)


def _norm_rope_store_torch(data, qkv, cfg):
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = qkv.shape[0]
    x = qkv.float()
    q = x[:, :hq * d].reshape(T, hq, d)
    k = x[:, hq * d:(hq + hkv) * d].reshape(T, hkv, d)
    v = x[:, (hq + hkv) * d:(hq + 2 * hkv) * d].reshape(T, hkv, d)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    if cfg["rope_theta"]:
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, d, 2).float() / d))
        ang = data["positions"].float()[:, None] * inv
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]

        def rope(z):
            e, o = z[..., 0::2], z[..., 1::2]
            return torch.stack([e * cos - o * sin, e * sin + o * cos], -1).flatten(-2)
        q, k = rope(q), rope(k)
    q = q.to(torch.bfloat16)
    kf = data["k_cache"].view(-1, hkv, d)
    vf = data["v_cache"].view(-1, hkv, d)
    kf[data["cache_loc"]] = (k.to(torch.bfloat16).float() / data["k_scale"]).to(FP8)
    vf[data["cache_loc"]] = (v / data["v_scale"]).to(FP8)
    return q.reshape(T, hq * d)


def _nsplit(cfg, batch, progs_per_split, capacity, occ=4):
    """Splits per request: ~occ programs per SM, slices of at least ~256 keys."""
    span = (cfg["window"] + cfg["q_len"] - 1) if cfg["window"] else capacity
    want = max(1, (132 * occ) // (batch * progs_per_split))
    by_len = max(1, span // 256)
    n = min(want, by_len, 256)
    return 1 << (n.bit_length() - 1)  # power of two (the combine's tl.arange)


# (layer kind, q_len, batch) -> (kernel, BLOCK_N, warps, stages[, programs per SM, hoist V load]);
# missing = SGLang's FA3 path.
# "head": one KV head per program; "rows": both KV heads per program (whole cache rows).
# Each entry beat FA3 on its shape in H200 check runs (v1b, v2b, v3, v4, v6 sweep A, v7 sweep B).
ATTN_CFG = {
    ("sliding", 1, 1): ("head", 64, 4, 3),
    ("sliding", 1, 16): ("head", 64, 4, 3),
    ("sliding", 1, 32): ("head", 64, 4, 3),
    ("sliding", 8, 16): ("head", 64, 4, 3),
    ("full", 1, 1): ("rows", 64, 4, 3),
    # rows + ~16 programs/SM + V load hoisted: 160 / 296 us vs 178 / 349 (head, 4/SM) in sweeps A, B
    ("full", 1, 16): ("rows", 64, 4, 3, 16, True),
    ("full", 1, 32): ("rows", 64, 4, 3, 16, True),
}
# Shapes not in the scored table (the tests) use the split-KV kernel with these defaults.
SCORED_BATCHES = (1, 16, 32)


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


def kernel_fn(data):
    cfg = data["config"]
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = batch * L
    qkv = data["qkv"].view(T, -1)
    q = _norm_rope_store(data, qkv, cfg)
    if q.device.type == "cpu":
        q = q.float()  # the Triton interpreter has no bf16
    kind = "sliding" if cfg["window"] else "full"
    key = (kind, L, batch)
    occ, hoist = 4, False
    if key in ATTN_CFG:
        which, block_n, warps, stages, *extra = ATTN_CFG[key]
        if extra:
            occ, hoist = extra
    elif batch in SCORED_BATCHES or q.device.type == "cpu":
        which, block_n, warps, stages = ("head", 64, 4, 3) if q.device.type == "cpu" else (None, 0, 0, 0)
    else:
        which, block_n, warps, stages = ("rows" if (hkv * d <= 128 and not cfg["window"] and L == 1) else "head",
                                         64, 4, 3)
    if which is None:
        return _fa3(data, q, qkv, cfg)
    if which == "rows" and hkv * d > 128:
        which = "head"
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
            HW=hw, HOIST=hoist, num_warps=warps, num_stages=stages)
    else:
        _attn_split[(hkv, nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"], BLOCK_M=block_m, BLOCK_N=block_n, D=d,
            HW=hw, HOIST=hoist, num_warps=warps, num_stages=stages)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = torch.empty((T, hq * d), device=qkv.device, dtype=data["qkv"].dtype)
    # one warp per (token, head) is latency-bound once it has >= 64 partials to read (7 us at 256)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=4 if nsplit >= 64 else 1)
    return out.view(batch, L, hq * d)
