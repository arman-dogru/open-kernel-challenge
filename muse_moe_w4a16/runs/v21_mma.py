# okc: problem=muse_moe_w4a16
"""v21: small-batch routed experts (bs <= SMALL_MAX) on tensor cores: a hand-written CUDA grouped
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
- GEMM 1 fuses gate and up (two MMAs per k-step on the same B fragment) into FP32 sums; GEMM 2
  stages silu(gate) * up rounded to BF16 (as SGLang) as its B operand; the FP32 post-norm +
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
__device__ __forceinline__ u32 bmul2(u32 a, u32 s2) {
  u32 r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(s2)); return r;
}
__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even (finite inputs)
  u32 u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (u16)(u >> 16);
}
// one E4M3 scale byte, given twice in the low 16 bits -> bf16x2 splat of (scale * 2^119), exact
// (both halves come from the same byte, so the cvt's half order does not matter)
__device__ __forceinline__ u32 scale_splat(u32 b16) {
  u32 h;
  asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h) : "r"(b16));
  u32 v = (h & 0x80008000u) | (((h >> 3) & 0x0FFF0FFFu) + 0x73807380u);
  return v & __vcmpne2(h & 0x7FFF7FFFu, 0u);
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

// grid (N / 16 / WARPS, work bound, K / KS); block 32 * WARPS.
// w: u32 tiles [32][ROWS/16][K/64][32 lanes][4]; sf: bytes [32][ROWS/16][K/64][8 g][4 s][2 h].
// out: FP32 [routes, OUTW], accumulated with atomics (zeroed by the caller). GATED: ROWS = 2N, gate sums
// at columns [0, N), up sums at [N, 2N), OUTW = 2N. Else ROWS = OUTW = N.
extern "C" __global__ void __launch_bounds__(32 * WARPS) moe_mma(
    const void* __restrict__ xin, const uint4* __restrict__ w, const uint2* __restrict__ sf,
    const float* __restrict__ gs, float* __restrict__ out, const int* __restrict__ cnt,
    const int* __restrict__ lst, const int* __restrict__ work, const int* __restrict__ nwork,
    int maxm, int local_start, int N, int K, int xrow_div) {
  __shared__ __align__(16) u16 xs[8 * XS];
  const int wk = blockIdx.y;
  if (wk >= *nwork) return;
  const int item = work[wk];
  const int ex = item / maxm, i0 = item % maxm;
  const int ne = cnt[ex];
  const int k0 = blockIdx.z * KS;
  // stage the chunk's 8 activation rows for [k0, k0 + KS) as BF16 (zeros past the chunk)
  for (int idx = threadIdx.x; idx < 8 * (KS / 8); idx += blockDim.x) {
    int m = idx / (KS / 8), c = idx % (KS / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (i0 + m < ne) {
      int r = lst[ex * maxm + i0 + m];
#if GATED
      v = reinterpret_cast<const uint4*>(reinterpret_cast<const u16*>(xin) + (size_t)(r / xrow_div) * K + k0)[c];
#else
      const float* hrow = reinterpret_cast<const float*>(xin) + (size_t)r * (2 * K) + k0 + c * 8;
      float4 g0 = reinterpret_cast<const float4*>(hrow)[0], g1 = reinterpret_cast<const float4*>(hrow)[1];
      float4 u0 = reinterpret_cast<const float4*>(hrow + K)[0], u1 = reinterpret_cast<const float4*>(hrow + K)[1];
      float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      float uu[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
      u32 pk[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float a = gg[2 * j] / (1.f + __expf(-gg[2 * j])) * uu[2 * j];
        float b = gg[2 * j + 1] / (1.f + __expf(-gg[2 * j + 1])) * uu[2 * j + 1];
        pk[j] = (u32)f2bf(a) | ((u32)f2bf(b) << 16);
      }
      v = make_uint4(pk[0], pk[1], pk[2], pk[3]);
#endif
    }
    reinterpret_cast<uint4*>(xs + m * XS)[c] = v;
  }
  __syncthreads();
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
#if GATED
  const size_t tu = tg + (size_t)RB * KB;
  const uint4* wu = w + tu * 32 + lane;
  const uint2* su = sf + tu * 8 + g;
  float cu[4] = {0.f, 0.f, 0.f, 0.f};
#endif
  float cg[4] = {0.f, 0.f, 0.f, 0.f};
  const u16* xb = xs + g * XS + 2 * t;
#pragma unroll 4
  for (int kb = 0; kb < KS / 64; kb++) {
    uint4 q = __ldg(wg + kb * 32);
    uint2 s = __ldg(sg + kb * 8);
#if GATED
    uint4 qu = __ldg(wu + kb * 32);
    uint2 su2 = __ldg(su + kb * 8);
#endif
#pragma unroll
    for (int st = 0; st < 4; st++) {
      const u16* xp = xb + kb * 64 + st * 16;
      u32 b0 = *reinterpret_cast<const u32*>(xp);
      u32 b1 = *reinterpret_cast<const u32*>(xp + 8);
      u32 wq = st == 0 ? q.x : st == 1 ? q.y : st == 2 ? q.z : q.w;
      u32 sw = st < 2 ? s.x : s.y;
      u32 slo = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x22 : 0x00));  // row g
      u32 shi = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x33 : 0x11));  // row g + 8
      u32 a0 = bmul2(rotl(wq, 6) & 0x81C081C0u, slo);
      u32 a1 = bmul2(rotl(wq, 2) & 0x81C081C0u, shi);
      u32 a2 = bmul2(rotl(wq, 28) & 0x81C081C0u, slo);
      u32 a3 = bmul2((wq & 0x81808180u) | (rotl(wq, 24) & 0x00400040u), shi);
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(cg[0]), "+f"(cg[1]), "+f"(cg[2]), "+f"(cg[3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#if GATED
      u32 wv = st == 0 ? qu.x : st == 1 ? qu.y : st == 2 ? qu.z : qu.w;
      u32 swu = st < 2 ? su2.x : su2.y;
      u32 ulo = scale_splat(__byte_perm(swu, 0, st & 1 ? 0x22 : 0x00));
      u32 uhi = scale_splat(__byte_perm(swu, 0, st & 1 ? 0x33 : 0x11));
      u32 e0 = bmul2(rotl(wv, 6) & 0x81C081C0u, ulo);
      u32 e1 = bmul2(rotl(wv, 2) & 0x81C081C0u, uhi);
      u32 e2 = bmul2(rotl(wv, 28) & 0x81C081C0u, ulo);
      u32 e3 = bmul2((wv & 0x81808180u) | (rotl(wv, 24) & 0x00400040u), uhi);
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(cu[0]), "+f"(cu[1]), "+f"(cu[2]), "+f"(cu[3])
                   : "r"(e0), "r"(e1), "r"(e2), "r"(e3), "r"(b0), "r"(b1));
#endif
    }
  }
  const float gsc = gs[local_start + ex] * 128.f;  // undoes the dequant's 2^-7
#if GATED
  const int OUTW = 2 * N;
#else
  const int OUTW = N;
#endif
  const int row = rb * 16 + g;
#pragma unroll
  for (int c = 0; c < 4; c++) {
    int n = 2 * t + (c & 1);
    if (i0 + n < ne) {
      int r = lst[ex * maxm + i0 + n];
      int rr = row + (c >> 1) * 8;
      atomicAdd(out + (size_t)r * OUTW + rr, cg[c] * gsc);
#if GATED
      atomicAdd(out + (size_t)r * OUTW + N + rr, cu[c] * gsc);
#endif
    }
  }
}
"""

_MODS = {}


def _mod(WARPS, KS, GATED):
    import cupy as cp
    key = (WARPS, KS, GATED)
    if key not in _MODS:
        _MODS[key] = cp.RawModule(code=_CU_SRC, options=(
            "-std=c++17", f"-DWARPS={WARPS}", f"-DKS={KS}", f"-DGATED={GATED}"))
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
MB = 8          # routes per MMA (the m16n8k16 N side)
# ((WARPS, SPLIT) for GEMM 1, (WARPS, SPLIT) for GEMM 2) per scored token count.
SMALL_CFG = {1: ((4, 4), (4, 4)), 8: ((4, 4), (4, 4)), 16: ((4, 2), (4, 2)), 32: ((4, 2), (4, 2))}
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
    (W1, S1), (W2, S2) = cfg_override or _small_cfg(bs)
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
    m1 = _mod(W1, H // S1, 1)
    m2 = _mod(W2, I // S2, 0)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((1,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                     np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(MB)))
        m1.get_function("moe_mma")((I // 16 // W1, nwork, S1), (32 * W1,), (
            _p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(h32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H), np.int32(top_k)))
        m2.get_function("moe_mma")((H // 16 // W2, nwork, S2), (32 * W2,), (
            _p(h32), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(y32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I), np.int32(1)))
    out = torch.empty((bs, H), device=dev, dtype=x.dtype)
    _post_norm_sum_kernel[(bs,)](y32, state["norm_w"], data["topk_weights"].float().contiguous(), slot, out,
                                 POST_NORM_EPS, TOPK=top_k, H=H, num_warps=8)
    return out


def prepare(weights):
    start, count = LOCAL
    st = _prepare_marlin(weights)
    w13, w13_sf = _repack_bank(weights["w13"], weights["w13_sf"], start, count)
    w2, w2_sf = _repack_bank(weights["w2"], weights["w2_sf"], start, count)
    st.update({
        "w13": w13, "w13_sf": w13_sf, "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": w2, "w2_sf": w2_sf, "w2_gs": weights["w2_gs"].float().contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    })
    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    for (W1, S1), (W2, S2) in SMALL_CFG.values():  # compile the launch configs up front
        _mod(W1, H // S1, 1)
        _mod(W2, I // S2, 0)
    torch.cuda.synchronize()
    if BENCH:
        _bench(st, weights)
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    assert (cfg["local_start"], cfg["local_count"]) == LOCAL
    if data["hidden_states"].shape[0] <= SMALL_MAX:
        return _kernel_small(data, state)
    return _kernel_marlin(data, state)


# EXPLORATION ONLY (check runs): BENCH times launch configs against the Marlin path inside prepare()
# on synthetic routings and prints them; nothing measured is kept or used by kernel_fn (which reads
# only the fixed SMALL_CFG table). A submit sets BENCH = False.
BENCH = True
_BENCHED = []
BENCH_WS = [(4, 4), (4, 2), (8, 2), (2, 4), (4, 8), (8, 4), (2, 8), (2, 2), (8, 8)]


def _synthetic(bs, gen, dev, I, H):
    """Heavy-tailed expert popularity, top-8 of 256, busiest group of 32 rotated to 0-31."""
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
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
        for ws in BENCH_WS:
            if time.time() - t0 > 300:
                print("BENCH time cap", flush=True)
                return
            try:
                t, out = _time(lambda: _kernel_small(data, st, (ws, ws)), flush)
                err = float((out.float() - ref.float()).norm() / ref.float().norm())
                rows.append((t, ws, err))
            except Exception as ex:
                print(f"BENCH bs={bs} ws={ws} error {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        rows.sort()
        for t, ws, err in rows:
            print(f"BENCH bs={bs} W,S={ws} {t:.1f}us err_vs_marlin={err:.4f}", flush=True)
        if rows:
            _, best, _ = rows[0]
            for ws in BENCH_WS:
                t, _ = _time(lambda: _kernel_small(data, st, (best, ws)), flush)
                print(f"BENCH bs={bs} w13={best} w2={ws} {t:.1f}us", flush=True)
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
