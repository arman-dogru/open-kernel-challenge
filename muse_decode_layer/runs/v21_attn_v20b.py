# okc: problem=muse_decode_layer
"""v21: v20 + attention v20b (per-token norm/RoPE/store from 64 tokens; sliding verify head kernel with V hoisted
and next-tile page-index prefetch). See tools/build_decode_v21.py.

v20: v19 with
- full-decode attention at batch >= 2 (split-KV "rows" kernel, 257 us at decode 32 full in v19's profile) on 128-key
  tiles with the next tile's page indices loaded one iteration ahead (PREF): each iteration waits on one DRAM round
  trip (the K / V gather) instead of two (page index, then gather). Same math.
- review of #17: the LAYER_SMALL_MAX comment now says NB 2 from 81 tokens (nearest SMALL_CFG entry, ties to the
  smaller: 80 -> 32, 81 -> 128); the dead NB == 4 launch-bounds branch is removed (no SMALL_CFG entry uses NB 4
  since v18b). v19's profile .log is empty because the kernel prints nothing (BENCH code removed in v18, r10).

v19: v18b with
- the CUDA experts also at 256 tokens (verify 32), with NB 2 items (16 routes per expert chunk) like at 128. v18 tried
  NB 4 there and lost (115 registers, 2 CTAs/SM); NB 2 has 80 registers / 3 CTAs per SM. In v18b's profile Marlin
  took 424 / 450 us at verify 32. The item model (~25 experts, ~40 NB 2 items, vs ~30 items at 128 tokens, where
  both expert GEMMs took ~275 us) projects ~350-370 us.
- the weight / block-scale loads of the expert kernels with an L2 evict-first policy and no L1 allocation, as MoE v26e
  (board 1.539x) has them. Same values loaded.

v18b: v18 with the CUDA experts capped at 192 tokens (128-token verify 16 keeps NB 2; 256-token verify 32 back on
Marlin). v18's check 9566de62f7e94ed1 (1.333x): NB 2 at 128 tokens beat Marlin (verify 16 sliding 422 -> 405 us, full
619 -> 587 us vs v17) while NB 4 at 256 lost (verify 32 sliding 600 -> 673, full 971 -> 1025 us: 115 registers,
2 CTAs/SM).

v18: v17 with the CUDA tensor-core experts extended from <= 32 to <= 256 tokens (verify 16 / 32: 128 / 256 tokens,
which ran Marlin): NB 2 work items (16 routes per expert chunk) at 128 tokens and NB 4 (32 routes, split 8 so the
staged rows fit 48 KB of static shared memory) at 256. Projection (judge routing, 100 seeds; v26c's measured
bs 8/16/32 fit time = 25.5 + 7.86 us per 8-route item; NB 2 / 4 items cost ~1.10 / ~1.35-1.45x, v23's bench):
~259 us at 128 tokens and ~343-366 us at 256 vs Marlin's ~339 / ~469 us standalone. GATED occupancy caps scale with NB
(NB 2: 3 CTAs/SM, NB 4: 2). FASTSCALE now lives in the MoE state (r10 minor) instead of a module global. More than 256
tokens still run Marlin.

v17: v16 plus muse_moe_w4a16 v26c's launch-chain fusion in the CUDA expert path (check f479a34c48984e24, 1.497x
there vs v26a 1.457x), 6 -> 4 launches with bit-identical math: the FP32 sum buffers are torch.empty and route_prep
(one block per route) zeroes the local routes' GEMM-1 / GEMM-2 rows; silu_mul is folded into GEMM 1: its split-K
CTAs count arrivals per (work item, row block) in self-resetting int32 counters (state["moe_small"]["c1"], zeroed
in prepare, reset to 0 by the last CTA inside the same launch, so every call and CUDA-graph replay starts from
zeros), and the last CTA writes silu(gate) * up as BF16 with silu_mul's expression. The CUDA experts now also run
17-32 tokens (decode 32) with the standalone MoE's 32-token config (NB 1, 8 warps, split 4, prefetch 2): v15b had
them ~6-11 us behind Marlin there before v26a's splat (~-10 us at bs32) and v26c's chain (~-10 us).

v16: v14 plus muse_moe_w4a16 v26a's scale splat (check bbebbbdf181f4934, 1.457x there vs v24 1.380x) in the
CUDA expert kernels (moe_mma, small token counts): for a positive normal E4M3 block-scale byte b (0x08..0x7E),
bf16(e4m3(b) * 2^119) is exactly (b << 4) + (239 << 7) per half, so the splat is one PRMT + one IMAD instead of
cvt e4m3x2 -> f16 -> f32, * 2^119, cvt bf16x2. prepare() checks every local scale byte is in 0x08..0x7E and
otherwise compiles v14's cvt path (FASTSCALE 0). Dequantized values are bit-identical to v14's.

v14: v13 plus muse_attention_decode v15b's verify routing (check ab5f387ae32242ba, 1.376x there): 8-token
verify blocks run the split-KV "head" kernel (FP8 x FP8 Q.K, BF16 P.V, FP32 softmax) on sliding layers at any
batch and on full layers at batch 1; full-layer verify at batch >= 2 stays on FA3 (the head kernel was 0.67-0.69x
of it there). Cleanups: the combine's dead NSPLIT_P2 masking removed, with a power-of-two assert in _nsplit (r6),
the _nsplit docstring (r6 n3), the layer summary / "Everything else" line now name the fused q/k front end
(r7 n1), and the RoPE clamp comment argues from the sliding bound (r7 n2). Precision is unchanged.

v13: v12 plus muse_attention_decode v14's attention front end (r6-approved, submitted as v14b):
  * one fused Triton norm + interleaved RoPE + FP8 + cache-write program per (token, head) replaces
    FusedQKNorm, the K/V/q casts and store_cache (4-5 launches) on every path; the RoPE table read is
    clamped to the table (r6 n4);
  * the query comes out of it in FP8 E4M3 (one RNE, saturating fp32 -> E4M3 rounding, as FA3 itself
    casts), so the FA3 verify path takes it as is (no separate cast launch), and the split-KV decode
    kernels run Q.K^T as an FP8 x FP8 tensor-core dot on the raw cache bytes; P.V stays BF16, softmax
    and accumulation FP32.
Not carried from v14: its dead SGLang wrappers (_qk_norm, _norm_rope_store, _norm_rope_store_torch).
Cleanups: SMALL_MAX and SMALL_CFG[32] removed (dead: the layer's CUDA experts stop at 16 tokens);
r4 n6 (hidden_states alignment assert), r4 n9 (FP32 intermediates note).

v12: v10 (check d761340c8ec04009, 1.243x) plus review r4 m1: CuPy compiles a RawModule lazily, so
prepare() now calls get_function on every kernel of every launch config, and NVRTC compiles (and any
errors) happen in prepare, never inside kernel_fn. (v11, the Triton RMSNorm at every token count, 1.234x,
was within noise of v10 and is not carried.) No bench or timing code in this file (r4 m2).

v10: v9 routed from its judge numbers (check 62ed09f29fd74783): the CUDA experts only up to 16
tokens (at 32 agentic-spread tokens moe_mma took 127.5 us vs Marlin's 124.3), the pooler back on v8's
Triton at 17-32 tokens and the unpooler back on SGLang's dispatch above 8 tokens (pre-transposed cuBLAS
lost at 128 / 256 tokens in the layer). The pre-expert RMSNorm becomes one Triton program per token
(torch's vectorized_layer_norm took 6.0 us at 1 token). The combine's NSPLIT_P2 masking stays: _nsplit
returns powers of two, so it is dead there, but it keeps the kernel correct for any split count.

v9: v8 plus muse_moe_w4a16 v24's tensor-core CUDA experts (NVRTC through CuPy, in-process) at
<= 32 tokens (check d66cab4599544266: 1.775x / 1.398x / 1.368x / 1.100x at 1 / 8 / 16 / 32 tokens vs
SGLang's Marlin path), fed the already-shifted local expert ids (local_start -> 0). Marlin stays above
32 tokens. Projection routing from bf16 v17's judge numbers (febf8e4adb9c4e1a): pre-transposed cuBLAS
for the pooler at 17-32 tokens and the unpooler above 8 tokens. Audit nits: _nsplit docstring.

Muse decoder layer: SGLang's layer (baseline.py, kept below) with our kernels swapped in where
their own problems' H200 runs showed a win:

  q/k front end   one fused Triton norm + RoPE + FP8 q + K/V cache-write program (muse_attention_decode
                  v13/v14) instead of FusedQKNorm, the casts and store_cache, on every path
  attention       split-KV Triton decode kernel + combine fused with the per-head norm x gate
                  (muse_attention_decode), by regime (_attn_path): decode steps, sliding verify and
                  full verify at batch 1; FA3 for full-layer verify at batch >= 2
  o, pooler       swap-AB Triton GEMM up to 32 tokens (muse_bf16_proj); cuBLAS above
  experts         SGLang's Marlin path, moe_block_size re-tuned at 128 / 256 tokens (muse_moe_w4a16);
                  SGLang's fused post-expert norm + weighted-sum Triton kernel launched at every
                  token count (its wrapper falls back to ~10 PyTorch kernels below 32 tokens)
  router          BF16 hi + lo Triton split-K GEMM for 9-32 tokens (SGLang runs the same hi/lo
                  split above 8 tokens, as two cuBLAS GEMMs); FP32 F.linear up to 8 as SGLang

Everything else (the layer's other norms, routing, Marlin experts above 16 tokens, unpooler) is SGLang's,
unchanged.

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
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
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
    if PREF:  # the first tile's page indices; each iteration then loads the next tile's (decode v21, as the rows kernel)
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
def _norm_rope_store_kernel(qkv_ptr, stride_qkv, pos_ptr, cs_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                            q_ptr, eps, NPOS, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                            ROPE: tl.constexpr, SCALED: tl.constexpr):
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
            # r6 n4: clamp to the table (2^21 rows, as SGLang's). RoPE runs on sliding layers only, whose judge
            # positions stay <= ~300,007, so the clamp never changes a judge input.
            pos = tl.minimum(tl.load(pos_ptr + t).to(tl.int64), NPOS - 1)
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
                                q_ptr, eps, NPOS, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                                ROPE: tl.constexpr, SCALED: tl.constexpr, HP: tl.constexpr):
    """Program t: every head of token t as one [HP, D/2] tile (attention v19, used from TOK_MIN tokens); head h < HQ a
    query head, < HQ + HKV a key head, < HQ + 2*HKV a value head (rows above masked). Math as _norm_rope_store_kernel."""
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


TOK_MIN = 64  # tokens from which the per-token norm/RoPE/store kernel is used (attention v19b / v20b, decode v21)


def _norm_rope_store_fused(data, qkv, cfg, cos_sin):
    """attention v14's fused step: per-head RMSNorm, interleaved RoPE, q out in FP8 E4M3 (one RNE,
    saturating fp32 -> E4M3 rounding, as FA3's own cast), k (BF16-rounded, / k_scale) and v (/ v_scale)
    cast to E4M3 and written to their cache rows. Replaces FusedQKNorm + the casts + store_cache."""
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = qkv.shape[0]
    q = torch.empty((T, hq * d), device=qkv.device, dtype=FP8)
    rope = bool(cfg["rope_theta"])
    args = (qkv, qkv.stride(0), data["positions"], cos_sin, data["cache_loc"],
            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
            q, RMS_EPS, cos_sin.shape[0])
    if T >= TOK_MIN:
        _norm_rope_store_tok_kernel[(T,)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=cfg["kv_scale"] != 1.0,
                                          HP=triton.next_power_of_2(hq + 2 * hkv), num_warps=2)
    else:
        _norm_rope_store_kernel[(T, hq + 2 * hkv)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope,
                                                   SCALED=cfg["kv_scale"] != 1.0, num_warps=1)
    return q


@triton.jit
def _combine_norm_gate(o_ptr, lse_ptr, gate_ptr, stride_gt, out_ptr, eps,
                       HQ: tl.constexpr, NSPLIT: tl.constexpr, D: tl.constexpr):
    """Merge the splits of one (token, head), then RMSNorm over the head x sigmoid(gate).
    NSPLIT is a power of two (_nsplit asserts it), so no lanes are masked."""
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
// streamed-once weight / scale loads: read-only path, no L1 allocation, L2 evict-first policy
// (one kernel-wide policy register: same registers / spills as without; as MoE v26e)
__device__ __forceinline__ uint4 ldw(const uint4* p, unsigned long long pol) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ uint2 lds2(const uint2* p, unsigned long long pol) {
  uint2 v;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(pol));
  return v;
}
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
#if FASTSCALE
// byte k of sw (positive normal E4M3) -> bf16x2 splat of e4m3 * 2^119, exact: (b << 4) + (239 << 7) per half
#define SPLAT(sw, k) (__byte_perm((sw), 0u, 0x4040u + 0x101u * (k)) * 16u + 0x77807780u)
#endif
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

// grid (NR): every block zeroes its route's FP32 GEMM-1 (RW1 floats) and GEMM-2 (RW2 floats) sum rows if the route
// is local; block 0 also builds the route lists.
extern "C" __global__ void route_prep(const int* __restrict__ ids, int* slot, int* cnt, int* lst, int* work,
                                      int* nwork, int NR, int local_start, int maxm, int mb,
                                      float* __restrict__ h, int RW1, float* __restrict__ y, int RW2) {
  {
    const int e = ids[blockIdx.x] - local_start;
    if (e >= 0 && e < 32) {
      float4* hr = reinterpret_cast<float4*>(h + (size_t)blockIdx.x * RW1);
      for (int c = threadIdx.x; c < RW1 / 4; c += blockDim.x) hr[c] = make_float4(0.f, 0.f, 0.f, 0.f);
      float4* yr = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * RW2);
      for (int c = threadIdx.x; c < RW2 / 4; c += blockDim.x) yr[c] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  if (blockIdx.x != 0) return;
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
#if FASTSCALE
    u32 slo = SPLAT(sw, st & 1 ? 2 : 0);  // row g
    u32 shi = SPLAT(sw, st & 1 ? 3 : 1);  // row g + 8
#else
    u32 slo = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x22 : 0x00));  // row g
    u32 shi = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x33 : 0x11));  // row g + 8
#endif
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
// GATED: keep v26a's CTAs / SM (8 warps: 4, 64 registers; 4 warps: 7, <= 73); the SiLU epilogue alone would push
// them to 70 / 83 registers.
#if GATED && WARPS == 8 && NB == 1
#define MINB 4
#elif GATED && WARPS == 4 && NB == 1
#define MINB 7
#elif GATED && WARPS == 8 && NB == 2
#define MINB 3
#else
#define MINB 1
#endif
extern "C" __global__ void __launch_bounds__(32 * WARPS, MINB) moe_mma(
    const u16* __restrict__ xin, const uint4* __restrict__ w, const uint2* __restrict__ sf,
    const float* __restrict__ gs, float* __restrict__ out, const int* __restrict__ cnt,
    const int* __restrict__ lst, const int* __restrict__ work, const int* __restrict__ nwork,
    int maxm, int local_start, int N, int K, int xrow_div, u16* __restrict__ inter, int* __restrict__ c1) {
  __shared__ __align__(16) u16 xs[8 * NB * XS];
  __shared__ int last;
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
  unsigned long long pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  for (int i = 0; i < PF; i++) if (i < NT) { qg[i] = ldw(wg + i * 32, pol); sgr[i] = lds2(sg + i * 8, pol); }
#if GATED
  const size_t tu = tg + (size_t)RB * KB;
  const uint4* wu = w + tu * 32 + lane;
  const uint2* su = sf + tu * 8 + g;
  uint4 qu[PF];
  uint2 sur[PF];
#pragma unroll
  for (int i = 0; i < PF; i++) if (i < NT) { qu[i] = ldw(wu + i * 32, pol); sur[i] = lds2(su + i * 8, pol); }
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
    if (kb + PF < NT) { qg[kb % PF] = ldw(wg + (kb + PF) * 32, pol); sgr[kb % PF] = lds2(sg + (kb + PF) * 8, pol); }
    mma_tile(cg, q, s, b0, b1);
#if GATED
    uint4 q2 = qu[kb % PF];
    uint2 s2 = sur[kb % PF];
    if (kb + PF < NT) { qu[kb % PF] = ldw(wu + (kb + PF) * 32, pol); sur[kb % PF] = lds2(su + (kb + PF) * 8, pol); }
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
#if GATED
  // split-K arrival count for (work item, row block); the last CTA writes silu(gate) * up as BF16 (silu_mul's math)
  __threadfence();
  __syncthreads();
  int* ctr = c1 + (size_t)wk * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) last = (atomicAdd(ctr, 1) == (int)gridDim.z - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int nr = min(8 * NB, ne - i0), r0 = blockIdx.x * WARPS * 16;
  for (int idx = threadIdx.x; idx < nr * WARPS * 16; idx += blockDim.x) {
    const int n = idx / (WARPS * 16), rr = r0 + idx % (WARPS * 16);
    const int r = lst[ex * maxm + i0 + n];
    const float gv = __ldcg(out + (size_t)r * OUTW + rr), uv = __ldcg(out + (size_t)r * OUTW + N + rr);
    inter[(size_t)r * N + rr] = f2bf(gv / (1.f + __expf(-gv)) * uv);
  }
  if (threadIdx.x == 0) *ctr = 0;
#endif
}
"""

_MODS = {}


def _scales_normal(sf, start, count):
    """True if every local block-scale byte is a positive normal E4M3 (0x08..0x7E): the SPLAT fast path is exact."""
    b = sf[start:start + count].contiguous().view(torch.uint8)
    return bool(((b >= 0x08) & (b <= 0x7E)).all())


def _mod(WARPS, KS, GATED, PF, NB, FAST):
    import cupy as cp
    key = (WARPS, KS, GATED, PF, NB, FAST)
    if key not in _MODS:
        _MODS[key] = cp.RawModule(code=_CU_SRC, options=(
            "-std=c++17", f"-DWARPS={WARPS}", f"-DKS={KS}", f"-DGATED={GATED}", f"-DPF={PF}", f"-DNB={NB}", f"-DFASTSCALE={FAST}"))
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


# (NB, (WARPS, SPLIT, PREFETCH) for GEMM 1, (WARPS, SPLIT, PREFETCH) for GEMM 2) per scored token count;
# a work item is up to 8 * NB routes of one expert (NB m16n8k16 MMAs per k-step on the same A fragment).
# Picked from v23's in-prepare bench (check b109e9378bc14d66); one config per scored token count.
# 32: the standalone MoE's 32-token entry (v23 bench, judge routing; v15a's layer bench: 8 and 16 warps tied).
# 128 / 256 (v18): NB 2 / NB 4 items (16 / 32 routes per expert chunk) cut the item count (~36 -> 27 at 128 tokens,
# ~63 -> 30 at 256 on the judge's routing). NB 4 lost at 256 (v18) and was dropped; v19 runs NB 2 at 256 as well.
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2)), 256: (2, (8, 4, 2), (8, 4, 2))}


def _small_cfg(bs):
    return SMALL_CFG[min(SMALL_CFG, key=lambda b: (abs(b - bs), b))]


def _kernel_small(data, state, cfg_override=None):
    import cupy as cp
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    assert x.data_ptr() % 16 == 0, "moe_mma reads hidden_states as uint4"  # r4 n6 (metadata: graph-safe)
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
    buf = torch.empty((NR * (2 * I + H),), device=dev, dtype=torch.float32)  # local rows zeroed by route_prep
    # h32 / y32: FP32 intermediates (higher precision than SGLang's BF16 caches) (r4 n9)
    h32 = buf[:NR * 2 * I]
    y32 = buf[NR * 2 * I:].view(NR, H)
    inter = torch.empty((NR, I), device=dev, dtype=torch.bfloat16)
    m1 = _mod(W1, H // S1, 1, P1, NB, state["fast"])
    m2 = _mod(W2, I // S2, 0, P2, NB, state["fast"])
    c1 = state["c1"]
    assert nwork * (I // 16 // W1) <= c1.numel()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((NR,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                      np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(MB),
                                                      _p(h32), np.int32(2 * I), _p(y32), np.int32(H)))
        m1.get_function("moe_mma")((I // 16 // W1, nwork, S1), (32 * W1,), (
            _p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(h32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H), np.int32(top_k),
            _p(inter), _p(c1)))
        m2.get_function("moe_mma")((H // 16 // W2, nwork, S2), (32 * W2,), (
            _p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(y32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I), np.int32(1),
            np.uint64(0), np.uint64(0)))
    out = torch.empty((bs, H), device=dev, dtype=x.dtype)
    _post_norm_sum_kernel[(bs,)](y32, state["norm_w"], data["topk_weights"].float().contiguous(), slot, out,
                                 POST_NORM_EPS, TOPK=top_k, H=H, num_warps=8)
    return out


# ---- end of v24 copy ----


PROJ_CFG = {"o": (64, 128, 1, 4, 4), "pooler": (32, 256, 1, 4, 4), "router": (16, 256, 8, 4, 4)}
PROJ_MAX_TOKENS = 32
# Marlin moe_block_size per token count, measured on muse_moe_w4a16 (SGLang picks 48 / 64)
MOE_BLOCK_M_OVERRIDE = {128: 16, 256: 32}
# Attention path: muse_attention_decode v15b's _attn_path, verbatim.
# Path by regime (layer kind, q_len, batch range), never by exact scored batch size, so nearby
# shapes take the same path as the scored ones (H200 evidence: check runs v1b..v8, see STATUS.md):
#   sliding decode (q_len 1), any batch      split-KV "head" kernel: 1.25-1.35x FA3 at b 1/16/32
#   full decode, batch 1                     split-KV "rows", ~4 programs/SM (256 splits): ~1.1x
#   full decode, batch >= 2                  "rows", ~16 programs/SM, V load hoisted: 1.17-1.26x
#                                            at b 16/32 (the 4/SM "head" config was 0.92x there)
#   sliding verify (q_len 8), any batch      split-KV "head" kernel (FP8 Q.K): 1.15-1.22x FA3
#   full verify, batch 1                     split-KV "head" kernel: 1.36x FA3
#   full verify, batch >= 2                  SGLang's FA3 (the head kernel was 0.67-0.69x at b16/32;
#                                            attention checks 28ad328cf5284948 / ab5f387ae32242ba)
# Value: (kernel, BLOCK_N, warps, stages, programs per SM, hoist V load); None = FA3.
def _attn_path(kind, L, batch, hkv, d):
    if L != 1:
        if hkv * d > 128 or (kind == "full" and batch >= 2):
            return None
        if kind == "sliding":  # decode v21 (attention v20b): V hoisted + next-tile page-index prefetch
            return ("head", 64, 4, 3, 4, True)
        return ("head", 64, 4, 3, 4, False)
    if kind == "sliding":
        return ("head", 64, 4, 3, 4, False)
    if hkv * d > 128:  # other head layouts (e.g. "dp"): not tuned, keep FA3
        return None
    if batch == 1:
        return ("rows", 64, 4, 3, 4, False)
    return ("rows", 128, 4, 3, 16, True)  # 128-key tiles + page-index prefetch (PREF = hoist), v20 / v17

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
    """Splits per request, a power of two in 1-256: ~occ programs per SM and at most ~span / 256 splits before
    rounding. Rounding to the nearest power of two can go up, so slices can be shorter than 256 keys (capacity
    1800 -> 8 splits of 225), never below ~128."""
    span = (cfg["window"] + cfg["q_len"] - 1) if cfg["window"] else capacity
    want = max(1, (132 * occ) // (batch * progs_per_split))
    by_len = max(1, span // 256)
    # Nearest power of two (audit r2 m5: flooring left batch 17 at ~8 programs/SM; exact counts
    # like 132 measured slower than 128 at b16, check 85e63326). The combine relies on a power of two.
    n = min(want, by_len, 256)
    p = 1 << (n.bit_length() - 1)
    r = min(2 * p, 256) if 2 * p - n < n - p else p
    assert r & (r - 1) == 0
    return r


def _splitkv_attention(data, q, qkv, cfg, gate, which, block_n, warps, stages, occ, hoist):
    """q [T, Hq*64] FP8 E4M3 after norm/RoPE (cache already written) -> rmsnorm(attn) * sigmoid(gate), [T, Hq*64]."""
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
            HW=True, HOIST=hoist, PREF=hoist, PV16=True, num_warps=warps, num_stages=stages)
    else:
        _attn_split[(hkv, nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"],
            BLOCK_M=max(16, triton.next_power_of_2(G * L)), BLOCK_N=block_n, D=d, HW=True, HOIST=hoist, PREF=hoist,
            PV16=True, num_warps=warps, num_stages=stages)
    out = torch.empty((T, hq * d), device=q.device, dtype=torch.bfloat16)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, gate.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
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


LAYER_SMALL_MAX = 256  # tokens up to this use the CUDA experts (v17: <= 32; v19: <= 256, NB 2 from 81 tokens)


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, N: tl.constexpr):
    """out = bf16(fp32 x * rsqrt(mean(x^2) + eps) * (1 + w)), as the reference's _rms(p) * (1 + w)."""
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / N + eps)
    w = tl.load(w_ptr + offs)
    tl.store(out_ptr + row * N + offs, (x * r * w).to(tl.bfloat16))


def _rmsnorm(x, w1):
    x = x.contiguous()
    out = torch.empty_like(x)
    _rmsnorm_kernel[(x.shape[0],)](x, w1, out, RMS_EPS, N=x.shape[1], num_warps=8)
    return out


def _moe_small_bank(weights):
    """v24's repacked local expert bank (all 32 experts of this GPU) and its compiled launch configs."""
    w13, w13_sf = _repack_bank(weights["w13"], weights["w13_sf"], 0, LOCAL_EXPERTS)
    w2, w2_sf = _repack_bank(weights["w2"], weights["w2_sf"], 0, LOCAL_EXPERTS)
    st = {"c1": torch.zeros((8192,), device=w13.device, dtype=torch.int32),  # GEMM-1 split-K counters (self-resetting)
          "w13": w13, "w13_sf": w13_sf, "w13_gs": weights["w13_gs"].float().contiguous(),
          "w2": w2, "w2_sf": w2_sf, "w2_gs": weights["w2_gs"].float().contiguous(),
          "norm_w": (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16)}
    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    # FASTSCALE per state (r10): the PRMT + IMAD splat only when every local scale byte is a positive normal E4M3
    st["fast"] = fast = int(_scales_normal(weights["w13_sf"], 0, LOCAL_EXPERTS)
                            and _scales_normal(weights["w2_sf"], 0, LOCAL_EXPERTS))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile now: CuPy is lazy (r4 m1)
        m1 = _mod(W1, H // S1, 1, P1, NB, fast)
        for name in ("route_prep", "moe_mma"):
            m1.get_function(name)
        _mod(W2, I // S2, 0, P2, NB, fast).get_function("moe_mma")
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
        "pre_expert_w1": (weights["pre_expert_norm"].float() + 1.0).contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "router_hi": hi.contiguous(),
        "router_lo": lo.contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
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
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    # One fused launch: norms, RoPE, FP8 q, K/V cache rows (slot 0 is a live row, reserved_skip_index=-1).
    q = _norm_rope_store_fused(data, qkv, cfg, state["cos_sin"])
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
            q=q.view(tokens, hq, d),  # already E4M3
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
    if tokens <= LAYER_SMALL_MAX:
        moe_data = {"hidden_states": _rmsnorm(expert_input, state["pre_expert_w1"]),
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
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
