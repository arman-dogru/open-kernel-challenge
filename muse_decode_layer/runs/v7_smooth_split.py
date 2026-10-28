# okc: problem=muse_decode_layer
"""Muse decoder layer: SGLang's layer (baseline.py, kept below) with our kernels swapped in where
their own problems' H200 runs showed a win:

  attention       split-KV Triton decode kernel + combine fused with the per-head norm x gate
                  (muse_attention_decode), on decode steps by regime (_attn_path); FA3 on verify
  o, pooler       swap-AB Triton GEMM up to 32 tokens (muse_bf16_proj); cuBLAS above
  experts         SGLang's Marlin path, moe_block_size re-tuned at 128 / 256 tokens (muse_moe_w4a16);
                  SGLang's fused post-expert norm + weighted-sum Triton kernel launched at every
                  token count (its wrapper falls back to ~10 PyTorch kernels below 32 tokens)
  router          BF16 hi + lo Triton split-K GEMM for 9-32 tokens (SGLang runs the same hi/lo
                  split above 8 tokens, as two cuBLAS GEMMs); FP32 F.linear up to 8 as SGLang

Everything else (norms, routing, Marlin experts, unpooler) is SGLang's, unchanged.

--- SGLang baseline notes ---
SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import triton
import triton.language as tl

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


# ---- our kernels -----------------------------------------------------------------------------
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
                       HQ: tl.constexpr, NSPLIT: tl.constexpr, D: tl.constexpr, NSPLIT_P2: tl.constexpr):
    """Merge the splits of one (token, head), then RMSNorm over the head x sigmoid(gate).
    NSPLIT need not be a power of two: NSPLIT_P2 = next power of two, extra lanes masked."""
    t = tl.program_id(0)
    hh = tl.program_id(1)
    sp = tl.arange(0, NSPLIT_P2)
    spm = sp < NSPLIT
    d = tl.arange(0, D)
    idx = (t * HQ + hh) * NSPLIT + sp
    lse = tl.load(lse_ptr + idx, mask=spm, other=float("-inf"))
    mx = tl.max(lse, 0)
    w = tl.where(lse == float("-inf"), 0.0, tl.exp2(lse - mx))
    o = tl.load(o_ptr + idx[:, None] * D + d[None, :], mask=spm[:, None], other=0.0)
    a = tl.sum(o * w[:, None], 0) / tl.sum(w, 0)
    r = a * tl.rsqrt(tl.sum(a * a, 0) / D + eps)
    g = tl.load(gate_ptr + t * stride_gt + hh * D + d).to(tl.float32)
    res = r * tl.sigmoid(g)
    tl.store(out_ptr + t * (HQ * D) + hh * D + d, res.to(out_ptr.dtype.element_ty))


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



# (BN, BK, SPLIT_K, warps, stages): muse_bf16_proj v3 configs
PROJ_CFG = {"o": (64, 128, 1, 4, 4), "pooler": (32, 256, 1, 4, 4), "router": (16, 256, 8, 4, 4)}
PROJ_MAX_TOKENS = 32
# Marlin moe_block_size per token count, measured on muse_moe_w4a16 (SGLang picks 48 / 64)
MOE_BLOCK_M_OVERRIDE = {128: 16, 256: 32}
# Attention path: muse_attention_decode's _attn_path (v9_regime), verbatim.
# Path by regime (layer kind, q_len, batch range), never by exact scored batch size, so nearby
# shapes take the same path as the scored ones (H200 evidence: check runs v1b..v8, see STATUS.md):
#   sliding decode (q_len 1), any batch      split-KV "head" kernel: 1.25-1.35x FA3 at b 1/16/32
#   full decode, batch 1                     split-KV "rows", ~4 programs/SM (256 splits): ~1.1x
#   full decode, batch >= 2                  "rows", ~16 programs/SM, V load hoisted: 1.17-1.26x
#                                            at b 16/32 (the 4/SM "head" config was 0.92x there)
#   any verify block (q_len 8)               SGLang's FA3: our 64-128-row blocks lose to it
#                                            on full layers (0.35-0.8x) and only tie or win by
#                                            <8% on sliding ones, so FA3 is kept at every batch
# Value: (kernel, BLOCK_N, warps, stages, programs per SM, hoist V load); None = FA3.
def _attn_path(kind, L, batch, hkv, d):
    if L != 1:
        return None
    if kind == "sliding":
        return ("head", 64, 4, 3, 4, False)
    if hkv * d > 128:  # other head layouts (e.g. "dp"): not tuned, keep FA3
        return None
    if batch == 1:
        return ("rows", 64, 4, 3, 4, False)
    return ("rows", 64, 4, 3, 16, True)

def _proj(x, w, layer, lo=None):
    x = x.contiguous()
    m, k = x.shape
    n = w.shape[0]
    bn, bk, split, warps, stages = PROJ_CFG[layer]
    bm = 16 if m <= 16 else 32
    if split > 1:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _swapab_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm), split)](
        x, w, w if lo is None else lo, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_LO=lo is not None, OUT_BF16=split == 1,
        num_warps=warps, num_stages=stages)
    return out


def _nsplit(cfg, batch, progs_per_split, capacity, occ=4):
    """Splits per request: ~occ programs per SM, slices of at least ~256 keys."""
    span = (cfg["window"] + cfg["q_len"] - 1) if cfg["window"] else capacity
    want = max(1, (132 * occ) // (batch * progs_per_split))
    by_len = max(1, span // 256)
    # No power-of-two rounding (audit r2 m5: flooring left batch 17 at ~8 programs/SM); the
    # combine masks up to the next power of two.
    return min(want, by_len, 256)


def _splitkv_attention(data, q, qkv, cfg, gate, which, block_n, warps, stages, occ, hoist):
    """q [T, Hq*64] after norm/RoPE (cache already written) -> rmsnorm(attn) * sigmoid(gate), [T, Hq*64]."""
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = batch * L
    G = hq // hkv
    page = cfg["page_size"]
    capacity = data["page_table"].shape[1] * page
    nsplit = _nsplit(cfg, batch, 1 if which == "rows" else hkv, capacity, occ)
    span = (cfg["window"] + L - 1) if cfg["window"] else capacity
    chunk = triton.cdiv(span, nsplit)
    o_part = torch.empty((T * hq * nsplit, d), device=q.device, dtype=torch.float32)
    lse = torch.empty((T * hq * nsplit,), device=q.device, dtype=torch.float32)
    args = (q, q.stride(0), data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8),
            data["page_table"], data["page_table"].stride(0), data["cache_seqlens"], data["k_scale"],
            data["v_scale"], o_part, lse, cfg["sm_scale"] * cfg["qk_scale_factor"], chunk, page)
    if which == "rows":
        rows = max(16, triton.next_power_of_2(hq * L))
        _attn_split_rows[(nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"], BLOCK_M=rows, BLOCK_N=block_n, D=d,
            HW=True, HOIST=hoist, num_warps=warps, num_stages=stages)
    else:
        _attn_split[(hkv, nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"],
            BLOCK_M=max(16, triton.next_power_of_2(G * L)), BLOCK_N=block_n, D=d, HW=True, HOIST=hoist,
            num_warps=warps, num_stages=stages)
    out = torch.empty((T, hq * d), device=q.device, dtype=torch.bfloat16)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, gate.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d, NSPLIT_P2=triton.next_power_of_2(nsplit),
                                num_warps=4 if nsplit >= 64 else 1)
    return out


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "router_hi": hi.contiguous(),
        "router_lo": lo.contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    block_size_m = MOE_BLOCK_M_OVERRIDE.get(num_tokens, block_size_m)
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _post_norm_sum(y, topk_weights, state["post_expert_w"], local_ids)


def _post_norm_sum(y, topk_weights, norm_w, slot_ids):
    """SGLang's fused post-expert norm + weighted sum kernel, launched directly at any token count:
    fused_post_expert_norm_weighted_sum's own launch, minus its < 32-token fallback (~10 PyTorch
    kernels doing the same math)."""
    tokens, topk, hidden = y.shape
    x = y.contiguous()
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x, norm_w, topk_weights.contiguous(), out, slot_ids,
        TOPK=topk, H=hidden, eps=POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True,
        HAS_IDS=True, BLOCK_H=block_h, num_warps=8 if block_h >= 4096 else 4, num_stages=2)
    return out


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    kind = "sliding" if window else "full"
    path = _attn_path(kind, length, batch, hkv, d)
    if path is not None:
        o = _splitkv_attention(data, q.reshape(tokens, hq * d), qkv, cfg, gate, *path)
    else:
        descale = {}
        if scaled:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
        out = flash_attn_with_kvcache(
            q=q.view(tokens, hq, d).to(FP8),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            scheduler_metadata=data["scheduler_metadata"],
            **descale,
        )
        if isinstance(out, tuple):
            out = out[0]
        o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    if tokens <= PROJ_MAX_TOKENS:
        attn_output = _proj(o, dense["o"], "o")  # all-reduce: identity
    else:
        attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    if tokens <= PROJ_MAX_TOKENS:
        expert_input = _proj(ffn_input, dense["pooler"], "pooler")
    else:
        expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router and tokens <= PROJ_MAX_TOKENS:
            logits = _proj(ffn_input, state["router_hi"], "router", lo=state["router_lo"])
        elif split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
