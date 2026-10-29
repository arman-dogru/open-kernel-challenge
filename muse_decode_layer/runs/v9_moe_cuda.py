# okc: problem=muse_decode_layer
"""v9: v8 plus muse_moe_w4a16 v24's tensor-core CUDA experts (NVRTC through CuPy, in-process) at
<= 32 tokens (check d66cab4599544266: 1.775x / 1.398x / 1.368x / 1.100x at 1 / 8 / 16 / 32 tokens vs
SGLang's Marlin path), fed the already-shifted local expert ids (local_start -> 0). Marlin stays above
32 tokens. Projection routing from bf16 v17's judge numbers (febf8e4adb9c4e1a): pre-transposed cuBLAS
for the pooler at 17-32 tokens and the unpooler above 8 tokens. Audit nits: _nsplit docstring.

Muse decoder layer: SGLang's layer (baseline.py, kept below) with our kernels swapped in where
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

import numpy as np
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
# ---- muse_moe_w4a16 v24 (tensor-core CUDA experts), copied verbatim from its runs/v24_tuned.py ----
# Where code j's bits (bit0 = mantissa, bit1/bit2 = exponent, bit3 = sign) go in the permuted
# 32-bit word. Pairs (0,1), (2,3), (4,5) come out of rotl(w, 6 / 2 / 28) & 0x81C081C0; pair (6,7)
# is (w & 0x81808180) | (rotl(w, 24) & 0x00400040).
_BITPOS = (
    (0, 1, 2, 9), (16, 17, 18, 25),      # rotl 6
    (4, 5, 6, 13), (20, 21, 22, 29),     # rotl 2
    (10, 11, 12, 19), (26, 27, 28, 3),   # rotl 28
    (14, 7, 8, 15), (30, 23, 24, 31),    # in place + mantissa from rotl 24
)


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



_CU_SRC = r"""
// Grouped small-batch NVFP4 x BF16 GEMM for routed experts on tensor cores (mma.sync m16n8k16, FP32
// accumulate), compiled in-process by NVRTC. Defines: WARPS (row blocks of 16 per CTA), KS (K range per
// CTA = K / split), GATED (1: x = BF16 hidden rows; 0: x = silu(gate) * up from the FP32 GEMM-1 sums).
typedef unsigned short u16;
typedef unsigned int u32;

__device__ __forceinline__ u32 rotl(u32 w, int n) { return __funnelshift_l(w, w, n); }
__device__ __forceinline__ u32 bmul2(u32 a, u32 s2) {
  u32 r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(s2)); return r;
}
__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even (finite inputs)
  u32 u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (u16)(u >> 16);
}
// one E4M3 scale byte (given twice in the low 16 bits, so the cvt's half order does not matter) ->
// bf16x2 splat of (scale * 2^119): e4m3 -> f16 -> f32 are exact, * 2^119 is exact (<= 4 significant bits,
// at most 448 * 2^119 < 2^128), and the f32 -> bf16 pack is exact for the same reason.
__device__ __forceinline__ u32 scale_splat(u32 b16) {
  u32 r;
  asm("{.reg .b16 t, lo, hi; .reg .b32 h; .reg .f32 f;\n"
      " cvt.u16.u32 t, %1;\n"
      " cvt.rn.f16x2.e4m3x2 h, t;\n"
      " mov.b32 {lo, hi}, h;\n"
      " cvt.f32.f16 f, lo;\n"
      " mul.f32 f, f, 0f7B000000;\n"
      " cvt.rn.bf16x2.f32 %0, f, f;}" : "=r"(r) : "r"(b16));
  return r;
}

extern "C" __global__ void route_prep(const int* __restrict__ ids, int* slot, int* cnt, int* lst, int* work,
                                      int* nwork, int NR, int local_start, int maxm, int mb) {
  __shared__ int scnt[32];
  int t = threadIdx.x;
  if (t < 32) scnt[t] = 0;
  __syncthreads();
  for (int r = t; r < NR; r += blockDim.x) {
    int e = ids[r] - local_start;
    if (e >= 0 && e < 32) {
      int pos = atomicAdd(&scnt[e], 1);
      lst[e * maxm + pos] = r;
    } else {
      e = -1;
    }
    slot[r] = e;
  }
  __syncthreads();
  if (t < 32) {
    int c = scnt[t];
    cnt[t] = c;
    int nch = (c + mb - 1) / mb;
    int incl = nch;
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
      if (t >= o) incl += v;
    }
    int base = incl - nch;
    for (int i = 0; i < nch; i++) work[base + i] = t * maxm + i * mb;
    if (t == 31) *nwork = incl;
  }
}

#define XS (KS + 8)  // padded smem row (bank-conflict-free B fragment loads)
#define NT (KS / 64)  // 16 x 64 tiles per warp

extern "C" __global__ void silu_mul(const float* __restrict__ h, u16* __restrict__ inter,
                                    const int* __restrict__ slot, int I) {
  const int r = blockIdx.x;
  if (slot[r] < 0) return;
  const float* hr = h + (size_t)r * 2 * I;
  for (int c = threadIdx.x * 4; c < I; c += blockDim.x * 4) {
    float4 g = *reinterpret_cast<const float4*>(hr + c), u = *reinterpret_cast<const float4*>(hr + I + c);
    float a0 = g.x / (1.f + __expf(-g.x)) * u.x, a1 = g.y / (1.f + __expf(-g.y)) * u.y;
    float a2 = g.z / (1.f + __expf(-g.z)) * u.z, a3 = g.w / (1.f + __expf(-g.w)) * u.w;
    uint2 v = make_uint2((u32)f2bf(a0) | ((u32)f2bf(a1) << 16), (u32)f2bf(a2) | ((u32)f2bf(a3) << 16));
    *reinterpret_cast<uint2*>(inter + (size_t)r * I + c) = v;
  }
}

__device__ __forceinline__ void mma_tile(float (*c)[4], uint4 q, uint2 s, u32 (*b0)[4], u32 (*b1)[4]) {
#pragma unroll
  for (int st = 0; st < 4; st++) {
    u32 wq = st == 0 ? q.x : st == 1 ? q.y : st == 2 ? q.z : q.w;
    u32 sw = st < 2 ? s.x : s.y;
    u32 slo = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x22 : 0x00));  // row g
    u32 shi = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x33 : 0x11));  // row g + 8
    u32 a0 = bmul2(rotl(wq, 6) & 0x81C081C0u, slo);
    u32 a1 = bmul2(rotl(wq, 2) & 0x81C081C0u, shi);
    u32 a2 = bmul2(rotl(wq, 28) & 0x81C081C0u, slo);
    u32 a3 = bmul2((wq & 0x81808180u) | (rotl(wq, 24) & 0x00400040u), shi);
#pragma unroll
    for (int nb = 0; nb < NB; nb++)
      asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[nb][0]), "+f"(c[nb][1]), "+f"(c[nb][2]), "+f"(c[nb][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[nb][st]), "r"(b1[nb][st]));
  }
}

// grid (N / 16 / WARPS, work bound, K / KS); block 32 * WARPS.
// xin: BF16 rows (row = route / xrow_div), K wide. w: u32 tiles [32][ROWS/16][K/64][32 lanes][4];
// sf: bytes [32][ROWS/16][K/64][8 g][4 s][2 h]. out: FP32 [routes, OUTW], accumulated with atomics (zeroed by
// the caller). GATED: ROWS = 2N, gate sums at columns [0, N), up sums at [N, 2N), OUTW = 2N; else ROWS = OUTW = N.
// The first PF tiles' weights are requested before the activation staging and barrier; the loop keeps PF tiles
// of loads in flight.
extern "C" __global__ void __launch_bounds__(32 * WARPS) moe_mma(
    const u16* __restrict__ xin, const uint4* __restrict__ w, const uint2* __restrict__ sf,
    const float* __restrict__ gs, float* __restrict__ out, const int* __restrict__ cnt,
    const int* __restrict__ lst, const int* __restrict__ work, const int* __restrict__ nwork,
    int maxm, int local_start, int N, int K, int xrow_div) {
  __shared__ __align__(16) u16 xs[8 * NB * XS];
  const int wk = blockIdx.y;
  if (wk >= *nwork) return;
  const int item = work[wk];
  const int ex = item / maxm, i0 = item % maxm;
  const int ne = cnt[ex];
  const int k0 = blockIdx.z * KS;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x * WARPS + warp;  // 16-row block (of the gate rows when GATED)
  const int KB = K / 64;
  const int RB = N / 16;
#if GATED
  const int RBT = 2 * RB;
#else
  const int RBT = RB;
#endif
  const size_t tg = ((size_t)ex * RBT + rb) * KB + k0 / 64;
  const uint4* wg = w + tg * 32 + lane;
  const uint2* sg = sf + tg * 8 + g;
  uint4 qg[PF];
  uint2 sgr[PF];
#pragma unroll
  for (int i = 0; i < PF; i++) if (i < NT) { qg[i] = __ldg(wg + i * 32); sgr[i] = __ldg(sg + i * 8); }
#if GATED
  const size_t tu = tg + (size_t)RB * KB;
  const uint4* wu = w + tu * 32 + lane;
  const uint2* su = sf + tu * 8 + g;
  uint4 qu[PF];
  uint2 sur[PF];
#pragma unroll
  for (int i = 0; i < PF; i++) if (i < NT) { qu[i] = __ldg(wu + i * 32); sur[i] = __ldg(su + i * 8); }
  float cu[NB][4];
#pragma unroll
  for (int nb = 0; nb < NB; nb++) cu[nb][0] = cu[nb][1] = cu[nb][2] = cu[nb][3] = 0.f;
#endif
  float cg[NB][4];
#pragma unroll
  for (int nb = 0; nb < NB; nb++) cg[nb][0] = cg[nb][1] = cg[nb][2] = cg[nb][3] = 0.f;
  // stage the chunk's 8 * NB activation rows for [k0, k0 + KS) as BF16 (zeros past the chunk)
  for (int idx = threadIdx.x; idx < 8 * NB * (KS / 8); idx += blockDim.x) {
    int m = idx / (KS / 8), c = idx % (KS / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (i0 + m < ne) {
      int r = lst[ex * maxm + i0 + m];
      v = reinterpret_cast<const uint4*>(xin + (size_t)(r / xrow_div) * K + k0)[c];
    }
    reinterpret_cast<uint4*>(xs + m * XS)[c] = v;
  }
  __syncthreads();
  const u16* xb = xs + g * XS + 2 * t;
#pragma unroll
  for (int kb = 0; kb < NT; kb++) {
    u32 b0[NB][4], b1[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; nb++)
#pragma unroll
      for (int st = 0; st < 4; st++) {
        b0[nb][st] = *reinterpret_cast<const u32*>(xb + nb * 8 * XS + kb * 64 + st * 16);
        b1[nb][st] = *reinterpret_cast<const u32*>(xb + nb * 8 * XS + kb * 64 + st * 16 + 8);
      }
    uint4 q = qg[kb % PF];
    uint2 s = sgr[kb % PF];
    if (kb + PF < NT) { qg[kb % PF] = __ldg(wg + (kb + PF) * 32); sgr[kb % PF] = __ldg(sg + (kb + PF) * 8); }
    mma_tile(cg, q, s, b0, b1);
#if GATED
    uint4 q2 = qu[kb % PF];
    uint2 s2 = sur[kb % PF];
    if (kb + PF < NT) { qu[kb % PF] = __ldg(wu + (kb + PF) * 32); sur[kb % PF] = __ldg(su + (kb + PF) * 8); }
    mma_tile(cu, q2, s2, b0, b1);
#endif
  }
  const float gsc = gs[local_start + ex] * 128.f;  // undoes the dequant's 2^-7
#if GATED
  const int OUTW = 2 * N;
#else
  const int OUTW = N;
#endif
  const int row = rb * 16 + g;
#pragma unroll
  for (int nb = 0; nb < NB; nb++)
#pragma unroll
    for (int c = 0; c < 4; c++) {
      int n = nb * 8 + 2 * t + (c & 1);
      if (i0 + n < ne) {
        int r = lst[ex * maxm + i0 + n];
        int rr = row + (c >> 1) * 8;
        atomicAdd(out + (size_t)r * OUTW + rr, cg[nb][c] * gsc);
#if GATED
        atomicAdd(out + (size_t)r * OUTW + N + rr, cu[nb][c] * gsc);
#endif
      }
    }
}
"""

_MODS = {}


def _mod(WARPS, KS, GATED, PF, NB):
    import cupy as cp
    key = (WARPS, KS, GATED, PF, NB)
    if key not in _MODS:
        _MODS[key] = cp.RawModule(code=_CU_SRC, options=(
            "-std=c++17", f"-DWARPS={WARPS}", f"-DKS={KS}", f"-DGATED={GATED}", f"-DPF={PF}", f"-DNB={NB}"))
    return _MODS[key]


def _p(t):
    return np.uint64(t.data_ptr())



def _repack_codes(w):
    """uint8 [ROWS, K/2] packed E2M1 (low nibble = even k), one expert -> uint8 tiles in mma.sync A-fragment
    order: [ROWS/16][K/64][lane = g*4+t][kstep s][8 codes j = kh*4 + h*2 + e] (code at row 16rb + 8h + g,
    k = 64kb + 16s + 8kh + 2t + e), two codes per byte; permute_codes then moves bits within each 32-bit word."""
    rows, k2 = w.shape
    k = 2 * k2
    nib = torch.stack([w & 15, w >> 4], -1).reshape(rows, k)
    t = nib.reshape(rows // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(-1, 8)
    return (t[:, 0::2] | (t[:, 1::2] << 4)).reshape(-1, 16).contiguous()


def _repack_scales(sf):
    """uint8 [ROWS, K/16] E4M3 bytes, one expert -> [ROWS/16][K/64][g][s][h] (row 16rb + 8h + g, group 4kb + s)."""
    rows, kg = sf.shape
    return sf.reshape(rows // 16, 2, 8, kg // 4, 4).permute(0, 3, 2, 4, 1).contiguous().reshape(-1)


def _repack_bank(w, sf, start, count):
    codes = torch.stack([permute_codes(_repack_codes(w[e])).view(torch.uint8).reshape(-1)
                         for e in range(start, start + count)])
    scales = torch.stack([_repack_scales(sf[e].contiguous().view(torch.uint8)) for e in range(start, start + count)])
    return codes.contiguous(), scales.contiguous()


SMALL_MAX = 32  # token counts up to this use the tensor-core CUDA path; larger ones use Marlin (v9b)
# (NB, (WARPS, SPLIT, PREFETCH) for GEMM 1, (WARPS, SPLIT, PREFETCH) for GEMM 2) per scored token count;
# a work item is up to 8 * NB routes of one expert (NB m16n8k16 MMAs per k-step on the same A fragment).
# Picked from v23's in-prepare bench (check b109e9378bc14d66); one config per scored token count.
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2))}


def _small_cfg(bs):
    return SMALL_CFG[min(SMALL_CFG, key=lambda b: (abs(b - bs), b))]


def _kernel_small(data, state, cfg_override=None):
    import cupy as cp
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    bs, H = x.shape
    I = cfg["d_expert"]
    top_k = cfg["top_k"]
    start, count = cfg["local_start"], cfg["local_count"]
    assert count == 32
    NB, (W1, S1, P1), (W2, S2, P2) = cfg_override or _small_cfg(bs)
    MB = 8 * NB
    NR = bs * top_k
    MAXM = NR
    nwork = min(NR, NR // MB + min(count, NR))
    dev = x.device
    ib = torch.empty((NR + count + count * MAXM + nwork + 1,), device=dev, dtype=torch.int32)
    slot = ib[:NR]
    cnt = ib[NR:NR + count]
    lst = ib[NR + count:NR + count + count * MAXM]
    work = ib[NR + count + count * MAXM:NR + count + count * MAXM + nwork]
    nw = ib[NR + count + count * MAXM + nwork:]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    buf = torch.zeros((NR * (2 * I + H),), device=dev, dtype=torch.float32)
    h32 = buf[:NR * 2 * I]
    y32 = buf[NR * 2 * I:].view(NR, H)
    inter = torch.empty((NR, I), device=dev, dtype=torch.bfloat16)
    m1 = _mod(W1, H // S1, 1, P1, NB)
    m2 = _mod(W2, I // S2, 0, P2, NB)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((1,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                     np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(MB)))
        m1.get_function("moe_mma")((I // 16 // W1, nwork, S1), (32 * W1,), (
            _p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(h32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H), np.int32(top_k)))
        m1.get_function("silu_mul")((NR,), (256,), (_p(h32), _p(inter), _p(slot), np.int32(I)))
        m2.get_function("moe_mma")((H // 16 // W2, nwork, S2), (32 * W2,), (
            _p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(y32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I), np.int32(1)))
    out = torch.empty((bs, H), device=dev, dtype=x.dtype)
    _post_norm_sum_kernel[(bs,)](y32, state["norm_w"], data["topk_weights"].float().contiguous(), slot, out,
                                 POST_NORM_EPS, TOPK=top_k, H=H, num_warps=8)
    return out


# ---- end of v24 copy ----


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
    """Splits per request: ~occ programs per SM, slices of about 256 keys. Rounding to the nearest
    power of two can go up, so slices can be shorter than 256 keys (capacity 1800 -> 8 splits of 225)."""
    span = (cfg["window"] + cfg["q_len"] - 1) if cfg["window"] else capacity
    want = max(1, (132 * occ) // (batch * progs_per_split))
    by_len = max(1, span // 256)
    # Nearest power of two (audit r2 m5: flooring left batch 17 at ~8 programs/SM; exact counts
    # like 132 measured slower than 128 at b16, check 85e63326). The combine masks either way.
    n = min(want, by_len, 256)
    p = 1 << (n.bit_length() - 1)
    return min(2 * p, 256) if 2 * p - n < n - p else p


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


def _moe_small_bank(weights):
    """v24's repacked local expert bank (all 32 experts of this GPU) and its compiled launch configs."""
    w13, w13_sf = _repack_bank(weights["w13"], weights["w13_sf"], 0, LOCAL_EXPERTS)
    w2, w2_sf = _repack_bank(weights["w2"], weights["w2_sf"], 0, LOCAL_EXPERTS)
    st = {"w13": w13, "w13_sf": w13_sf, "w13_gs": weights["w13_gs"].float().contiguous(),
          "w2": w2, "w2_sf": w2_sf, "w2_gs": weights["w2_gs"].float().contiguous(),
          "norm_w": (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16)}
    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():
        _mod(W1, H // S1, 1, P1, NB)
        _mod(W2, I // S2, 0, P2, NB)
    return st


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
        "pooler_t": weights["pooler"].t().contiguous(),
        "unpooler_t": weights["unpooler"].t().contiguous(),
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
        "moe_small": _moe_small_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    state["moe_small"]["norm_w"] = state["post_expert_w"]  # the same effective (1 + w) the Marlin path uses
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
    if tokens <= 16:
        expert_input = _proj(ffn_input, dense["pooler"], "pooler")
    elif tokens <= PROJ_MAX_TOKENS:
        expert_input = torch.mm(ffn_input, state["pooler_t"])  # bf16 v17: 1.069x vs Triton 1.031x at 32
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
    if tokens <= SMALL_MAX:
        moe_data = {"hidden_states": state["pre_expert_norm"](expert_input),
                    "topk_ids": local_ids.to(torch.int32), "topk_weights": topk_weights,
                    "config": {"d_expert": D_EXPERT, "top_k": TOP_K, "local_start": 0,
                               "local_count": LOCAL_EXPERTS}}
        expert_output = _kernel_small(moe_data, state["moe_small"])
    else:
        expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = torch.mm(expert_output, state["unpooler_t"])  # bf16 v17: T 1.01-1.03x vs SGLang

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
