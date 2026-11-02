# okc: problem=muse_moe_w4a16
"""v27d2_bench = v26e + MoE v27 (tools/moe_v27_stream.cu): a weight-streaming grouped expert GEMM for <= 32 tokens.
v26e streams the touched experts at ~2.2-2.9 TB/s (moe_mma: per-warp register prefetch, split-K FP32 atomics).
v27: persistent grid (one CTA per SM), equal contiguous shares of the (segment x rowtile x k-chunk) stage sequence,
a producer warp streaming the A-fragment tiles with TMA bulk copies (L2 evict-first) into an mbarrier ring, the
segment's activations staged once in shared memory, no split-K atomics (the <= 2 units a CTA shares with its
neighbour are combined through a workspace + per-CTA launch counters; fixed order). GEMM 1 writes the BF16
SiLU product, GEMM 2 FP32 rows; route_prep (lists only, one block) and v26e's post-norm Triton kernel unchanged.
v27d: kernel_fn is v26e unchanged; prepare() runs a one-time untimed sweep of v27 tunes vs v26e (diagnostic),
plus NOCOMPUTE builds of the best tunes (consumers skip the dequant + MMAs: the pure weight-streaming time).

v26e = v26c with:
- the CUDA tensor-core experts extended from <= 32 to <= 192 tokens, NB 2 work items (16 routes per expert chunk) at
  128 tokens (verify 16; it ran Marlin). In the decode layer the same kernels at 128 tokens took verify 16 from
  422 / 619 us (v17, Marlin) to 405 / 587 us (v18 check 9566de62f7e94ed1) and 402 / 594 us (v18b ae89a962123748e6).
  256 tokens stay on Marlin (NB 4 lost there: 115 registers, 2 CTAs/SM).
- weight and block-scale loads with an L2 evict-first cache policy and no L1 allocation (createpolicy +
  ld.global.nc.L1::no_allocate.L2::cache_hint), as Marlin does, so the FP32 sums, routing lists and activations
  stay L2-resident while ~0.1-0.7 GB of weights stream through once. Same values loaded: bit-identical to v26c.
- review round 10 minors: FASTSCALE per state (no module global), eager NVRTC compile of every launch config in
  prepare() (RawModule.compile()), and the exploration-only BENCH code removed.

v26c = v26a with the small-batch launch chain cut from 6 kernels to 4 (bit-identical outputs):
- the FP32 sum buffers are torch.empty (no zero-fill kernel): route_prep runs one block per route and zeroes that
  route's GEMM-1 (2I) and GEMM-2 (H) FP32 rows if the route is local (block 0 also builds the lists, as before);
- silu_mul is gone: GEMM 1's split-K CTAs count arrivals per (work item, row block) in a self-resetting int32
  counter (state["c1"], zero at prepare, reset by its last CTA), and the last CTA of each (item, row block) applies
  silu(gate) * up to its 16 * WARPS rows of the item's routes and writes them as BF16, with silu_mul's expression.
Launches: route_prep, moe_mma (GEMM 1 + SiLU), moe_mma (GEMM 2), post-norm sum.

v26a = v24 with a 2-instruction E4M3 scale splat (FASTSCALE): for a positive normal E4M3 byte b
(0x08..0x7E), bf16(e4m3(b) * 2^119) is exactly (b << 4) + (239 << 7) (exponent e - 7 + 119 + 127 = e + 239, the
3 mantissa bits land in the top of bf16's 7). The splat is one PRMT (b into both 16-bit halves as 0x00bb00bb)
and one IMAD (* 16 + 0x77807780) instead of v23's cvt e4m3x2 -> f16 -> f32, * 2^119, cvt bf16x2 (~5); the
k-step does 8 splats per 16 x 64 tile, about a third of its instructions (moe_mma is issue-bound: NB = 2's extra
MMAs cost 8-11% in v23's bench). Exhaustively checked on CPU for all 119 bytes. prepare() checks that every
local block-scale byte is in 0x08..0x7E (judge weights: scales clamped to [8, 448], or >= 2^-6 with wide scales,
all normal) and otherwise compiles v24's exact cvt path (FASTSCALE 0). The stored weights are unchanged.

v24 = v23 with SMALL_CFG re-picked from v23's bench (NB = 1 everywhere: one 8-route MMA per
k-step beat NB = 2 / 4 at every token count) and BENCH = False (no bench in prepare).
v23 = v22 with: E4M3 scale -> bf16 splat by cvt e4m3x2 -> f16 -> f32, * 2^119, cvt.bf16x2 (5 ops instead
of ~14 integer ops; the scale path was most of the k-step's instructions), and NB m16n8k16 MMAs per k-step
(work items of up to 8 * NB routes of one expert share one pass over its weights).
v22 = v21 with: the first PF tiles' weight loads issued before the activation staging and barrier,
a PF-deep register ring of weight loads in a fully unrolled k loop, non-volatile mma asm, and GEMM 2
staging precomputed BF16 silu(gate) * up rows (one silu_mul kernel over local routes) instead of
recomputing it from the FP32 sums in every CTA.
v21: small-batch routed experts (bs <= SMALL_MAX) on tensor cores: a hand-written CUDA grouped
NVFP4 x BF16 GEMM (mma.sync m16n8k16, FP32 accumulate) compiled in-process with NVRTC through CuPy
(no subprocess) and launched on PyTorch's current stream (graph-capturable: no syncs, no allocation
in the launches); Marlin (the board's v9b path, unchanged) above SMALL_MAX tokens.

- prepare() repacks the 32 local experts' E2M1 codes (lossless) into mma A-fragment order: per 16-row
  x 64-k tile, each lane's 16 bytes hold its 4 k-steps (rows g and g + 8, k pairs 2t and 2t + 8), with
  v11-v20's fixed bit permutation inside each 32-bit word; the E4M3 scales are reordered to match
  (8 bytes per lane per tile). NVFP4 stays NVFP4; scales stay E4M3; global scales FP32.
- Dequant into the A fragment: 4 bf16x2 per 32-bit word by rotate + AND [+ OR], times the row's
  E4M3 scale (cvt.e4m3x2 -> f16x2 of the byte twice, re-biased to bf16 * 2^119 with integer ops, exact) via
  mul.rn.bf16x2: e2m1 * scale * 2^-7, exact. B fragment = up to 8 routes of one expert (BF16 rows
  staged in shared memory, padded rows: no bank conflicts). FP32 accumulation in the MMA; global
  scale * 2^7 in the epilogue; split-K partial sums added with FP32 atomics into a zeroed buffer.
- route_prep (one block): per-expert route lists, a compact work list of 8-route chunks, slot ids.
- GEMM 1 fuses gate and up (two MMAs per k-step on the same B fragment) into FP32 sums; silu_mul
  writes silu(gate) * up rounded to BF16 (as SGLang), GEMM 2's B operand; the FP32 post-norm +
  weighted sum (Triton) combines each token's local routes."""

from typing import TypedDict

import types

import numpy as np
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
// (one kernel-wide policy register: same registers / spills as v26c; making the policy inside each load spilled more)
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


SMALL_MAX = 192  # token counts up to this use the tensor-core CUDA path; larger ones use Marlin (v9b)
# (NB, (WARPS, SPLIT, PREFETCH) for GEMM 1, (WARPS, SPLIT, PREFETCH) for GEMM 2) per scored token count;
# a work item is up to 8 * NB routes of one expert (NB m16n8k16 MMAs per k-step on the same A fragment).
# Picked from v23's in-prepare bench (check b109e9378bc14d66); one config per scored token count.
# 128 (v26e): NB 2 items, 16 routes per expert chunk (decode v18 / v18b measured the gain at 128 tokens).
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2))}
LOCAL = (0, 32)


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
    buf = torch.empty((NR * (2 * I + H),), device=dev, dtype=torch.float32)  # local rows zeroed by route_prep
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


# ---------------- MoE v27 (tools/moe_v27_stream.cu, configs tools/moe_v27_cfg.py) ----------------
_CU27 = r"""// MoE v27: weight-streaming grouped NVFP4 x BF16 expert GEMM (routed experts, <= 32 tokens), sm_90a, mma.sync
// m16n8k16 (W rows = MMA M, the routes of one expert = N), compiled by NVRTC with -D GATED, NT, NW, ST, TPS, FASTSCALE.
// Why: v26e's moe_mma (grid rowblocks x work items x split-K, per-warp register prefetch of 2 tiles) streams the
// experts at ~2.2-2.9 TB/s. v27 applies bf16 v33's design:
// - persistent grid of one CTA per SM. Every CTA rebuilds the segment list from cnt[] (segment = up to TC = 8 NT routes
//   of one local expert) and takes an equal contiguous share of the global stage sequence
//   q = ((segment * RP + rowtile p) * SPU + k-chunk), SPU = K / 64 / TPS (a unit = segment x output rowtile p: for
//   GEMM 1 the gate rowtile p and the up rowtile p, for GEMM 2 rowtile p; full K).
// - one producer warp streams each stage's weight tiles (codes 512 B + E4M3 scales 64 B per 16 x 64 tile, the v26
//   A-fragment repack) with TMA bulk copies (L2 evict-first) into an ST-deep ring; NW consumer warps dequantize from
//   shared memory (v26a's rotate / AND / PRMT-IMAD splat / mul.bf16x2) and run the MMAs against the segment's
//   activation rows, staged once per segment in shared memory (padded rows, conflict-free B fragments).
// - no split-K atomics: a unit is finished by one CTA; the at most two units a CTA shares with a neighbour (its range
//   starts / ends inside a unit) are combined through a global workspace: the CTA owning the unit's start adds the next
//   CTA's partial after that CTA publishes a per-CTA launch counter (st.release.gpu by its only writer, all counters
//   of the grid in lockstep; ld.acquire.gpu). Fixed order (own + next): deterministic. GEMM 1's epilogue applies
//   silu(gate * gs) * (up * gs) and writes BF16 (v26c's expression); GEMM 2 writes FP32 rows y * gs.
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 rotl(u32 w, int n) { return __funnelshift_l(w, w, n); }
__device__ __forceinline__ u32 bmul2(u32 a, u32 s2) {
  u32 r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(s2)); return r;
}
__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even (finite inputs)
  u32 u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (u16)(u >> 16);
}
#if FASTSCALE
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
__device__ __forceinline__ u32 sa(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u64* b, u32 count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(u64* b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u64* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(u64* b, u32 parity) {  // traps after ~2^26 failed polls (never hang)
  u32 ok = 0, n = 0;
  while (true) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(sa(b)), "r"(parity) : "memory");
    if (ok) break;
    if (++n > (1u << 26)) asm volatile("trap;");
  }
}
__device__ __forceinline__ void bulk_ef(void* dst, const void* src, u32 bytes, u64* bar, u64 pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(sa(p))
               : "memory");
  return v;
}
__device__ __forceinline__ uint2 lds64(const void* p) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(sa(p)) : "memory");
  return v;
}
__device__ __forceinline__ void bar_consumers() { asm volatile("bar.sync 1, %0;" :: "r"(NW * 32) : "memory"); }
__device__ __forceinline__ void spin_eq(const u32* p, u32 want) {
  u32 v, n = 0;
  while (true) {
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    if (v == want) break;
    if (++n > (1u << 26)) asm volatile("trap;");
  }
}
__device__ __forceinline__ void st_release(u32* p, u32 v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// v26's 16 x 64 tile MMA: q = this lane's 16 code bytes (4 k-steps), s = its 8 scale bytes; b0 / b1 = B fragments
__device__ __forceinline__ void mma_tile(float (*c)[4], uint4 q, uint2 s, const u32 (*b0)[4], const u32 (*b1)[4]) {
#pragma unroll
  for (int st = 0; st < 4; st++) {
    u32 wq = st == 0 ? q.x : st == 1 ? q.y : st == 2 ? q.z : q.w;
    u32 sw = st < 2 ? s.x : s.y;
#if FASTSCALE
    u32 slo = SPLAT(sw, st & 1 ? 2 : 0);  // row g
    u32 shi = SPLAT(sw, st & 1 ? 3 : 1);  // row g + 8
#else
    u32 slo = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x22 : 0x00));
    u32 shi = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x33 : 0x11));
#endif
    u32 a0 = bmul2(rotl(wq, 6) & 0x81C081C0u, slo);
    u32 a1 = bmul2(rotl(wq, 2) & 0x81C081C0u, shi);
    u32 a2 = bmul2(rotl(wq, 28) & 0x81C081C0u, slo);
    u32 a3 = bmul2((wq & 0x81808180u) | (rotl(wq, 24) & 0x00400040u), shi);
#pragma unroll
    for (int nb = 0; nb < NT; nb++)
      asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[nb][0]), "+f"(c[nb][1]), "+f"(c[nb][2]), "+f"(c[nb][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[nb][st]), "r"(b1[nb][st]));
  }
}

#define TC (8 * NT)          // routes per segment
#define E (NT * 128)         // FP32 partials per (unit, half): 16 rows x TC routes
#define HALVES (GATED ? 2 : 1)
#define TB (TPS * 576)       // code + scale bytes per half per stage
#define SB (HALVES * TB)     // stage bytes
#define MAXSEG 128
#ifndef NOCOMPUTE
#define NOCOMPUTE 0  // diagnostic only: consumers skip the dequant + MMAs (pure weight-streaming time; wrong output)
#endif

// grid: P CTAs (one per SM), block (NW + 1) * 32, dynamic smem: ST * SB + TC * (K + 8) * 2 + NW * HALVES * E * 4
//   + (2 * ST) * 8. w: codes [32][RBT][K/64][512 B]; sf: scales [32][RBT][K/64][64 B] (RBT = HALVES * N / 16;
//   GATED: gate rowtiles [0, N/16), up rowtiles [N/16, 2N/16)). xin: BF16 rows (route r -> row r / xrow_div), K wide.
//   GATED: inter[r][N] = BF16 silu(gate) * up; else y[r][N] FP32. ws: [P][HALVES * E] FP32; ctr: [P] u32 counters.
extern "C" __global__ void __launch_bounds__((NW + 1) * 32, 1) moe_stream(
    const u16* __restrict__ xin, const unsigned char* __restrict__ w, const unsigned char* __restrict__ sf,
    const float* __restrict__ gs, const int* __restrict__ cnt, const int* __restrict__ lst, int maxm,
    int local_start, int N, int K, int xrow_div, u16* __restrict__ inter, float* __restrict__ y,
    float* __restrict__ ws, u32* __restrict__ ctr) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ int s_ex[MAXSEG], s_i0[MAXSEG], s_n[MAXSEG];
  __shared__ int s_nseg;
  __shared__ u32 s_old;
  const int XSR = K + 8;  // padded activation row (u16)
  unsigned char* ring = sm;
  u16* xs = reinterpret_cast<u16*>(sm + ST * SB);
  float* red = reinterpret_cast<float*>(sm + ST * SB + TC * XSR * 2);
  u64* bars = reinterpret_cast<u64*>(red + NW * HALVES * E);
  u64* full = bars;
  u64* empty = bars + ST;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int RP = N / 16, KB = K / 64, SPU = KB / TPS, RBT = HALVES * RP;
  if (warp == 0) {  // segment list from the per-expert route counts
    const int c = cnt[lane];
    const int nch = (c + TC - 1) / TC;
    int incl = nch;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
      if (lane >= o) incl += v;
    }
    for (int i = 0; i < nch; ++i) {
      const int sgi = incl - nch + i;
      if (sgi < MAXSEG) {
        s_ex[sgi] = lane;
        s_i0[sgi] = i * TC;
        s_n[sgi] = min(TC, c - i * TC);
      }
    }
    if (lane == 31) s_nseg = min(incl, MAXSEG);
    if (lane == 0) {
      for (int i = 0; i < ST; ++i) {
        mbar_init(full + i, 1);
        mbar_init(empty + i, NW);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      s_old = *reinterpret_cast<volatile u32*>(ctr + blockIdx.x);
    }
  }
  __syncthreads();
  const long long U = (long long)s_nseg * RP * SPU;
  const long long q0 = U * blockIdx.x / gridDim.x, q1 = U * (blockIdx.x + 1) / gridDim.x;
  const u32 want = s_old + 1u;
  // stage q -> (segment sg, rowtile p, k-chunk kc), advanced incrementally (no 64-bit div / mod per stage)
  const long long uq0 = q0 / SPU;
  const int kcq0 = (int)(q0 - uq0 * SPU), pq0 = (int)(uq0 % RP), sgq0 = (int)(uq0 / RP);
  if (warp == NW) {  // producer
    if (lane == 0) {
      u64 pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      int it = 0, kc = kcq0, p = pq0, sg = sgq0;
      for (long long q = q0; q < q1; ++q, ++it) {
        const int ex = s_ex[sg];
        const int s = it % ST;
        if (it >= ST) mbar_wait(empty + s, (u32)((it / ST) - 1) & 1u);
        mbar_expect_tx(full + s, SB);
        unsigned char* dst = ring + s * SB;
#pragma unroll
        for (int h = 0; h < HALVES; ++h) {
          const size_t t0 = ((size_t)ex * RBT + h * RP + p) * KB + (size_t)kc * TPS;  // first tile of this stage
          bulk_ef(dst + h * TPS * 512, w + t0 * 512, TPS * 512, full + s, pol);
          bulk_ef(dst + HALVES * TPS * 512 + h * TPS * 64, sf + t0 * 64, TPS * 64, full + s, pol);
        }
        if (++kc == SPU) {
          kc = 0;
          if (++p == RP) { p = 0; ++sg; }
        }
      }
    }
    return;
  }
  // consumers
  const int g = lane >> 2, t = lane & 3;
  float acc[HALVES][NT][4];
#pragma unroll
  for (int h = 0; h < HALVES; ++h)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[h][j][c] = 0.f;
  int cur = -1;
  bool published = false;
  int it = 0, kcn = kcq0, pn = pq0, sgn = sgq0;
  long long un = uq0;
  for (long long q = q0; q < q1; ++q, ++it) {
    const int kc = kcn, p = pn, sg = sgn;
    const long long u = un;
    if (++kcn == SPU) {
      kcn = 0;
      ++un;
      if (++pn == RP) { pn = 0; ++sgn; }
    }
    const int ex = s_ex[sg];
    if (sg != cur) {  // stage this segment's activation rows (BF16, K wide) once
      if (cur >= 0) bar_consumers();  // everyone is done with the previous rows
      const int n = s_n[sg], i0 = s_i0[sg];
      for (int idx = threadIdx.x; idx < n * (K / 8); idx += NW * 32) {
        const int m = idx / (K / 8), c = idx % (K / 8);
        const int r = lst[ex * maxm + i0 + m];
        reinterpret_cast<uint4*>(xs + m * XSR)[c] = __ldg(reinterpret_cast<const uint4*>(xin + (size_t)(r / xrow_div) * K) + c);
      }
      bar_consumers();
      cur = sg;
    }
    const int s = it % ST;
    mbar_wait(full + s, (u32)(it / ST) & 1u);
    const unsigned char* st0 = ring + s * SB;
    for (int i = warp; i < (NOCOMPUTE ? 0 : TPS); i += NW) {
      const int kb = kc * TPS + i;
      u32 b0[NT][4], b1[NT][4];
      const u16* xb = xs + g * XSR + 2 * t + kb * 64;
#pragma unroll
      for (int nb = 0; nb < NT; nb++)
#pragma unroll
        for (int k4 = 0; k4 < 4; k4++) {
          b0[nb][k4] = *reinterpret_cast<const u32*>(xb + nb * 8 * XSR + k4 * 16);
          b1[nb][k4] = *reinterpret_cast<const u32*>(xb + nb * 8 * XSR + k4 * 16 + 8);
        }
#pragma unroll
      for (int h = 0; h < HALVES; ++h) {
        const uint4 qc = lds128(st0 + h * TPS * 512 + i * 512 + lane * 16);
        const uint2 sc = lds64(st0 + HALVES * TPS * 512 + h * TPS * 64 + i * 64 + g * 8);
        mma_tile(acc[h], qc, sc, b0, b1);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + s);
    const bool uend = kc == SPU - 1, rend = q == q1 - 1;
    if (!(uend || rend)) continue;
    // unit (or this CTA's part of it) done: fixed-order sum of the NW warps' partials
    bar_consumers();  // previous readers of red are done
#pragma unroll
    for (int h = 0; h < HALVES; ++h)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          red[(warp * HALVES + h) * E + (j * 4 + c) * 32 + lane] = acc[h][j][c];
          acc[h][j][c] = 0.f;
        }
    bar_consumers();
    const long long ustart = u * SPU, uendq = ustart + SPU;
    const int tid = threadIdx.x;
    if (ustart < q0) {  // shared with the previous CTA, which finishes it: hand over the partial
      for (int e = tid; e < HALVES * E; e += NW * 32) {
        float v = 0.f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) v += red[(w2 * HALVES) * E + e];
        ws[(size_t)blockIdx.x * HALVES * E + e] = v;
      }
      __threadfence();
      bar_consumers();
      if (tid == 0) st_release(ctr + blockIdx.x, want);
      published = true;
      continue;
    }
    const bool shared_next = uendq > q1;  // the next CTA holds the rest; its partial is added here
    if (shared_next) {
      if (tid == 0) spin_eq(ctr + blockIdx.x + 1, want);
      bar_consumers();
    }
    const float gsc = gs[local_start + ex] * 128.f;  // undoes the dequant's 2^-7
    const int n = s_n[sg], i0 = s_i0[sg];
    for (int e = tid; e < E; e += NW * 32) {
      float v[HALVES];
#pragma unroll
      for (int h = 0; h < HALVES; ++h) {
        float a = 0.f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) a += red[(w2 * HALVES + h) * E + e];
        if (shared_next) a += __ldcg(ws + (size_t)(blockIdx.x + 1) * HALVES * E + h * E + e);
        v[h] = a * gsc;
      }
      const int ln = e & 31, c = (e >> 5) & 3, j = e >> 7;
      const int row = p * 16 + (ln >> 2) + 8 * (c >> 1);
      const int col = 8 * j + 2 * (ln & 3) + (c & 1);
      if (col < n) {
        const int r = lst[ex * maxm + i0 + col];
#if GATED
        inter[(size_t)r * N + row] = f2bf(v[0] / (1.f + __expf(-v[0])) * v[1]);
#else
        y[(size_t)r * N + row] = v[0];
#endif
      }
    }
  }
  if (!published) {
    bar_consumers();
    if (threadIdx.x == 0) st_release(ctr + blockIdx.x, want);
  }
}
"""

# """MoE v27 launch configs (shared by the runs files and the offline tests). Config = (GATED, NT, NW, ST, TPS)."""
V27_SMEM_LIMIT = 227 * 1024 - 2048  # dynamic bytes (the kernel has ~1.6 KB static)
V27_SMEM_MIN = 116 * 1024           # one CTA per SM


def v27_smem(cfg, K):
    gated, nt, nw, st, tps = cfg
    halves = 2 if gated else 1
    return st * halves * tps * 576 + 8 * nt * (K + 8) * 2 + nw * halves * nt * 128 * 4 + 2 * st * 8


def v27_valid(cfg, K=4096):
    gated, nt, nw, st, tps = cfg
    return (K // 64) % tps == 0 and tps % nw == 0 and v27_smem(cfg, K) <= V27_SMEM_LIMIT


# per GEMM: candidate (NW, ST, TPS) tunes
V27_TUNES = [(8, 4, 16), (8, 3, 32), (8, 4, 32), (16, 3, 32), (16, 4, 32), (8, 2, 64), (16, 2, 64), (16, 3, 64)]
V27_ALL = [(g, nt, nw, st, tps) for g in (1, 0) for nt in (1, 2) for (nw, st, tps) in V27_TUNES
               if v27_valid((g, nt, nw, st, tps))]

V27_ROUTE = {}
_MODS27 = {}


def _mod27(cfg, fast, K, nc=0):
    """cfg = (GATED, NT, NW, ST, TPS) -> (cupy function, dynamic smem bytes); compiled + attribute set once.
    nc = 1: NOCOMPUTE diagnostic build (weight streaming only, wrong output)."""
    import cupy as cp
    key = (cfg, fast, K, nc)
    if key not in _MODS27:
        g, nt, nw, st, tps = cfg
        assert v27_valid(cfg, K), cfg
        m = cp.RawModule(code=_CU27, options=("-std=c++17", f"-DGATED={g}", f"-DNT={nt}", f"-DNW={nw}", f"-DST={st}",
                                              f"-DTPS={tps}", f"-DFASTSCALE={fast}", f"-DNOCOMPUTE={nc}"))
        f = m.get_function("moe_stream")
        shm = max(v27_smem(cfg, K), V27_SMEM_MIN)  # >= 116 KB: one CTA per SM
        f.max_dynamic_shared_size_bytes = shm
        _MODS27[key] = (f, shm, nw)
    return _MODS27[key]


def _kernel_stream(data, state, t1, t2, stages=3, nc=0):
    """t1 / t2 = GEMM-1 / GEMM-2 tunes (NT, NW, ST, TPS). stages < 3: diagnostic partial paths (route_prep + GEMM 1 ...)."""
    import cupy as cp
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    bs, H = x.shape
    I = cfg["d_expert"]
    top_k = cfg["top_k"]
    start, count = cfg["local_start"], cfg["local_count"]
    assert count == 32 and H % 1024 == 0 and I % 1024 == 0
    NR = bs * top_k
    MAXM = NR
    dev = x.device
    ib = torch.empty((NR + count + count * MAXM + NR + 1,), device=dev, dtype=torch.int32)
    slot = ib[:NR]
    cnt = ib[NR:NR + count]
    lst = ib[NR + count:NR + count + count * MAXM]
    work = ib[NR + count + count * MAXM:NR + count + count * MAXM + NR]
    nw = ib[NR + count + count * MAXM + NR:]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    inter = torch.empty((NR, I), device=dev, dtype=torch.bfloat16)
    y32 = torch.empty((NR, H), device=dev, dtype=torch.float32)  # non-local rows never read (slot < 0)
    f1, s1, w1 = _mod27((1,) + tuple(t1), state["fast"], H, nc)
    f2, s2, w2 = _mod27((0,) + tuple(t2), state["fast"], I, nc)
    P = state["v27_P"]
    ws, ctr = state["v27_ws"], state["v27_ctr"]
    m1 = _mod(4, H // 8, 1, 2, 1, state["fast"])  # route_prep only (lists, slot ids; RW1 = RW2 = 0: no zeroing)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((1,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                     np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(8),
                                                     np.uint64(0), np.int32(0), np.uint64(0), np.int32(0)))
        if stages >= 2:
            f1((P,), (32 * (w1 + 1),), (_p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(cnt),
                                       _p(lst), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H),
                                       np.int32(top_k), _p(inter), np.uint64(0), _p(ws), _p(ctr)), shared_mem=s1)
        if stages >= 3:
            f2((P,), (32 * (w2 + 1),), (_p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(cnt),
                                       _p(lst), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I),
                                       np.int32(1), np.uint64(0), _p(y32), _p(ws), _p(ctr)), shared_mem=s2)
    out = torch.empty((bs, H), device=dev, dtype=x.dtype)
    if stages >= 3:
        _post_norm_sum_kernel[(bs,)](y32, state["norm_w"], data["topk_weights"].float().contiguous(), slot, out,
                                     POST_NORM_EPS, TOPK=top_k, H=H, num_warps=8)
    return out


def _v27_prepare(st, H, I):
    dev = st["w13"].device
    P = torch.cuda.get_device_properties(dev).multi_processor_count
    st["v27_P"] = P
    st["v27_ws"] = torch.empty((P * 2 * 256,), device=dev, dtype=torch.float32)  # shared-unit partials
    st["v27_ctr"] = torch.zeros((P,), device=dev, dtype=torch.int32)  # per-CTA launch counters (lockstep)
    for t1, t2 in V27_ROUTE.values():  # NVRTC-compile every routed config now
        _mod27((1,) + tuple(t1), st["fast"], H)
        _mod27((0,) + tuple(t2), st["fast"], I)


_DIAG27 = {}


def _v27_gen(bs, seed, dev):
    """reference.generate_input(bs, "ep8", seed, dist="normal", rank="busiest") routing, inlined (diagnostic only)."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def _v27_diag(state):
    """One-time untimed sweep per scored token count <= 32: v26e (as checked) vs v27 with every valid GEMM-1 tune
    (GEMM 2 default) then every GEMM-2 tune (best GEMM 1); CUDA graph replay, cold L2 (256 MB read first), median
    of 15; also route_prep + GEMM 1 alone and an empty-ish floor; rel L2 error of each v27 output vs v26e."""
    import time
    if _DIAG27 or not torch.cuda.is_available():
        return
    _DIAG27["done"] = True
    t0 = time.time()
    dev = state["w13"].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    tiny = torch.empty(16, device=dev)

    def timed(fn, reps=15):
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        ts = []
        for _ in range(reps):
            flush.view(torch.int32).sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    floor = timed(lambda: tiny.zero_())
    print(f"v27d2_bench diag: {torch.cuda.get_device_name()} SMs {state['v27_P']} fast {state['fast']} floor {floor:.2f} us",
          flush=True)
    EXP = 4096 * 4096 * 3 * 9 // 16  # bytes per expert (codes + scales), GEMM 1 is 2/3
    summ = []
    for bs in (1, 8, 16, 32):
        if time.time() - t0 > 420:
            print("(time guard)", flush=True)
            break
        data = _v27_gen(bs, 1000 + bs, dev)
        touched = int(torch.unique(data["topk_ids"][data["topk_ids"] < 32]).numel())
        ref = _kernel_small(data, state).float()
        t26 = timed(lambda: _kernel_small(data, state))
        nts = (1,) if bs <= 8 else (1, 2)
        res = []

        def run(t1, t2, stages=3, nc=0):
            try:
                o = _kernel_stream(data, state, t1, t2, stages, nc)
                if stages == 3 and not nc:
                    torch.cuda.synchronize()
                    err = float((o.float() - ref).norm() / ref.norm().clamp(min=1e-30))
                else:
                    err = 0.0
                t = timed(lambda: _kernel_stream(data, state, t1, t2, stages, nc))
                return t, err
            except Exception as e:
                print(f"  bs {bs} {t1} {t2} FAILED {repr(e)[:160]}", flush=True)
                return 1e9, 1.0

        d2 = (1, 8, 3, 32)
        g1 = []
        for nt in nts:
            for (nw, st, tps) in V27_TUNES:
                t1 = (nt, nw, st, tps)
                if v27_valid((1,) + t1, 4096) and v27_valid((0, nt) + d2[1:], 4096):
                    t, e = run(t1, (nt,) + d2[1:])
                    g1.append((t, e, t1))
        g1.sort()
        best1 = g1[0][2]
        g2 = []
        for (nw, st, tps) in V27_TUNES:
            t2 = (best1[0], nw, st, tps)
            if v27_valid((0,) + t2, 4096):
                t, e = run(best1, t2)
                g2.append((t, e, t2))
        g2.sort()
        best2 = g2[0][2]
        tg1, _ = run(best1, best2, 2)
        trp, _ = run(best1, best2, 1)
        tb = g2[0][0]
        nc1, _ = run(best1, best2, 2, 1)
        nc2, _ = run(best1, best2, 3, 1)
        bw1 = touched * EXP * 2 / 3 / max(tg1 - trp, 1e-3) / 1e6
        bw2 = touched * EXP / 3 / max(tb - tg1 - 3.0, 1e-3) / 1e6
        print(f"bs {bs}: experts {touched}, v26e {t26:.1f} us | v27 best {tb:.1f} us {best1} {best2} e{g2[0][1]:.1e}"
              f" | route_prep {trp:.1f}, +G1 {tg1:.1f} (~{bw1:.2f} TB/s G1, ~{bw2:.2f} TB/s G2 est)"
              f" | NOCOMPUTE: +G1 {nc1:.1f} (~{touched * EXP * 2 / 3 / max(nc1 - trp, 1e-3) / 1e6:.2f} TB/s), all {nc2:.1f}",
              flush=True)
        print("  G1 tunes: " + "; ".join(f"{c} {t:.1f} e{e:.0e}" for t, e, c in g1), flush=True)
        print("  G2 tunes: " + "; ".join(f"{c} {t:.1f} e{e:.0e}" for t, e, c in g2), flush=True)
        summ.append((bs, t26, tb, best1, best2))
    print("v27d summary (bs: v26e / v27 us, tunes): " + "; ".join(
        f"{b}: {a:.1f} / {c:.1f} {list(t1)} {list(t2)}" for b, a, c, t1, t2 in summ), flush=True)
    print(f"v27d2_bench diag: done in {time.time() - t0:.0f} s", flush=True)


def prepare(weights):
    start, count = LOCAL
    st = _prepare_marlin(weights)
    w13, w13_sf = _repack_bank(weights["w13"], weights["w13_sf"], start, count)
    w2, w2_sf = _repack_bank(weights["w2"], weights["w2_sf"], start, count)
    st.update({
        "c1": torch.zeros((8192,), device=weights["w13"].device, dtype=torch.int32),  # GEMM-1 split-K counters
        "w13": w13, "w13_sf": w13_sf, "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": w2, "w2_sf": w2_sf, "w2_gs": weights["w2_gs"].float().contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    })
    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    # FASTSCALE per state (r10): the PRMT + IMAD splat only when every local scale byte is a positive normal E4M3
    st["fast"] = fast = int(_scales_normal(weights["w13_sf"], start, count)
                            and _scales_normal(weights["w2_sf"], start, count))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # NVRTC-compile every launch config now (r10)
        _mod(W1, H // S1, 1, P1, NB, fast).compile()
        _mod(W2, I // S2, 0, P2, NB, fast).compile()
    _mod(4, H // 8, 1, 2, 1, fast).compile()  # v27's route_prep module
    _v27_prepare(st, H, I)
    torch.cuda.synchronize()
    try:
        _v27_diag(st)  # v27d: diagnostic only, untimed
    except Exception as e:  # never fail the run
        print("v27d diag failed:", repr(e)[:600], flush=True)
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    assert (cfg["local_start"], cfg["local_count"]) == LOCAL
    if data["hidden_states"].shape[0] <= SMALL_MAX:
        return _kernel_small(data, state)
    return _kernel_marlin(data, state)


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
