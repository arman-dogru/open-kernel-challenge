# okc: problem=muse_moe_w4a16
"""v20: small-batch routed experts (bs <= SMALL_MAX) as a hand-written CUDA grouped GEMV, compiled
in-process with NVRTC through CuPy (no subprocess) and launched on PyTorch's current stream (graph-
capturable: no syncs, no allocation inside the launches); Marlin (the board's v9b path, unchanged)
above SMALL_MAX tokens.

- prepare() keeps the 32 local experts' E2M1 codes as int64 words of 16 codes with a fixed lossless
  bit permutation inside each 32-bit half (as v11-v19); E4M3 scales and FP32 global scales untouched.
  NVFP4 stays NVFP4.
- route_prep (one block): per-expert route lists via shared-memory atomics, a compact work list of
  chunks of MB routes of one expert (warp prefix sum), slot ids for the post-norm sum.
- moe_gemv: block = WARPS warps x R rows each; the chunk's x rows staged in shared memory as BF16;
  each lane walks the K words (8-byte coalesced loads), dequantises 8 codes per 4 bf16x2 ops
  (rotate + AND [+ OR] then mul.rn.bf16x2 by scale * 2^119: exact e2m1 * scale * 2^-7), FP32 FMAs
  for all MB routes, warp-shuffle reduction, global scale * 2^7 in the epilogue. GEMM 1 fuses gate
  and up rows and stores silu(gate) * up as BF16 (as SGLang); GEMM 2 stores FP32 per route; the FP32
  post-norm + weighted sum (Triton) combines each token's local routes."""

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



def _permute_local(w, start, count, chunk=4):
    """Permuted int64 code words for experts [start, start + count) only."""
    out = torch.empty((count, w.shape[1], w.shape[2] // 8), dtype=torch.int64, device=w.device)
    for e in range(0, count, chunk):
        out[e:e + chunk] = permute_codes(w[start + e:start + min(e + chunk, count)])
    return out



_CU_SRC = r"""
// Grouped small-batch NVFP4 GEMV for routed experts (compiled in-process by NVRTC).
// Defines: MB (routes per work item, 1..4), R (rows per warp), WARPS, KMAX (max K), GATED (0/1).
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 rotl(u32 w, int n) { return __funnelshift_l(w, w, n); }

__device__ __forceinline__ u32 bmul2(u32 a, u32 s2) {  // bf16x2 multiply, round to nearest
  u32 r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(s2)); return r;
}

__device__ __forceinline__ float e4m3_to_f32(u32 b) {
  u32 h; asm("{.reg .b16 lo; cvt.u16.u32 lo, %1; cvt.rn.f16x2.e4m3x2 %0, lo;}" : "=r"(h) : "r"(b));
  float f; asm("{.reg .b16 l, hh; mov.b32 {l, hh}, %1; cvt.f32.f16 %0, l;}" : "=f"(f) : "r"(h));
  return f;
}

// 8 permuted E2M1 codes (one 32-bit half word) -> 8 floats (k order) = e2m1 * scale * 2^-7.
__device__ __forceinline__ void deq8(u32 w, u32 s2, float* o) {
  u32 p0 = bmul2(rotl(w, 6) & 0x81C081C0u, s2);
  u32 p1 = bmul2(rotl(w, 2) & 0x81C081C0u, s2);
  u32 p2 = bmul2(rotl(w, 28) & 0x81C081C0u, s2);
  u32 p3 = bmul2((w & 0x81808180u) | (rotl(w, 24) & 0x00400040u), s2);
  o[0] = __uint_as_float(p0 << 16); o[1] = __uint_as_float(p0 & 0xFFFF0000u);
  o[2] = __uint_as_float(p1 << 16); o[3] = __uint_as_float(p1 & 0xFFFF0000u);
  o[4] = __uint_as_float(p2 << 16); o[5] = __uint_as_float(p2 & 0xFFFF0000u);
  o[6] = __uint_as_float(p3 << 16); o[7] = __uint_as_float(p3 & 0xFFFF0000u);
}

__device__ __forceinline__ u32 scale2(u32 sbyte) {  // E4M3 scale * 2^119 as bf16x2 (exact)
  float s = e4m3_to_f32(sbyte) * 6.646139978924579e+35f;
  u32 b = __float_as_uint(s) >> 16;
  return b | (b << 16);
}

__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even
  u32 u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (u16)(u >> 16);
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

// grid: (N / (WARPS * R), work bound). block: 32 * WARPS.
// x: bf16 [*, K] (row = route / xrow_div). w: int64 words [32, ROWS, K/16]; sf: u8 [32, ROWS, K/16].
// GATED: ROWS = 2N (gate rows then up rows); out = bf16 [routes, N] = silu(gate) * up.
// else:  ROWS = N; out = fp32 [routes, N].
extern "C" __global__ void __launch_bounds__(32 * WARPS) moe_gemv(
    const u16* __restrict__ x, const u64* __restrict__ w, const unsigned char* __restrict__ sf,
    const float* __restrict__ gs, void* __restrict__ out, const int* __restrict__ cnt,
    const int* __restrict__ lst, const int* __restrict__ work, const int* __restrict__ nwork,
    int maxm, int local_start, int N, int K, int xrow_div) {
  __shared__ __align__(16) u16 xs[MB * KMAX];
  const int wk = blockIdx.y;
  if (wk >= *nwork) return;
  const int item = work[wk];
  const int ex = item / maxm, i0 = item % maxm;
  const int ne = cnt[ex];
  int route[MB];
#pragma unroll
  for (int m = 0; m < MB; m++) route[m] = (i0 + m < ne) ? lst[ex * maxm + i0 + m] : -1;
  // stage x rows (bf16) in shared memory
  const int K8 = K / 8;
  for (int idx = threadIdx.x; idx < MB * K8; idx += blockDim.x) {
    int m = idx / K8, c = idx % K8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (route[m] >= 0) v = reinterpret_cast<const uint4*>(x + (size_t)(route[m] / xrow_div) * K)[c];
    reinterpret_cast<uint4*>(xs + m * KMAX)[c] = v;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int KW = K / 16;
#if GATED
  const int ROWS = 2 * N;
#else
  const int ROWS = N;
#endif
  const int n0 = (blockIdx.x * WARPS + warp) * R;
  const u64* wb = w + ((size_t)ex * ROWS + n0) * KW;
  const unsigned char* sb = sf + ((size_t)ex * ROWS + n0) * KW;
  float acc[R][GATED + 1][MB];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int g = 0; g <= GATED; g++)
#pragma unroll
      for (int m = 0; m < MB; m++) acc[r][g][m] = 0.f;
#pragma unroll 2
  for (int kw = lane; kw < KW; kw += 32) {
    u64 q[R][GATED + 1];
    u32 sc[R][GATED + 1];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int g = 0; g <= GATED; g++) {
        size_t off = ((size_t)g * N + r) * KW + kw;
        q[r][g] = __ldg(wb + off);
        sc[r][g] = __ldg(sb + off);
      }
#pragma unroll
    for (int h = 0; h < 2; h++) {  // low / high 8 codes of the word
      float xv[MB][8];
#pragma unroll
      for (int m = 0; m < MB; m++) {
        uint4 v = reinterpret_cast<const uint4*>(xs + m * KMAX + kw * 16 + h * 8)[0];
        xv[m][0] = __uint_as_float(v.x << 16); xv[m][1] = __uint_as_float(v.x & 0xFFFF0000u);
        xv[m][2] = __uint_as_float(v.y << 16); xv[m][3] = __uint_as_float(v.y & 0xFFFF0000u);
        xv[m][4] = __uint_as_float(v.z << 16); xv[m][5] = __uint_as_float(v.z & 0xFFFF0000u);
        xv[m][6] = __uint_as_float(v.w << 16); xv[m][7] = __uint_as_float(v.w & 0xFFFF0000u);
      }
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int g = 0; g <= GATED; g++) {
          float o[8];
          u32 half = h ? (u32)(q[r][g] >> 32) : (u32)q[r][g];
          deq8(half, scale2(sc[r][g]), o);
#pragma unroll
          for (int m = 0; m < MB; m++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[r][g][m] = fmaf(o[j], xv[m][j], acc[r][g][m]);
        }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int g = 0; g <= GATED; g++)
#pragma unroll
      for (int m = 0; m < MB; m++) {
        float v = acc[r][g][m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        acc[r][g][m] = v;
      }
  if (lane == 0) {
    const float gsc = gs[local_start + ex] * 128.f;  // undoes the dequant's 2^-7
#pragma unroll
    for (int m = 0; m < MB; m++) {
      if (route[m] < 0) continue;
#pragma unroll
      for (int r = 0; r < R; r++) {
#if GATED
        float gv = acc[r][0][m] * gsc, uv = acc[r][1][m] * gsc;
        float sv = gv / (1.f + __expf(-gv)) * uv;
        reinterpret_cast<u16*>(out)[(size_t)route[m] * N + n0 + r] = f2bf(sv);
#else
        reinterpret_cast<float*>(out)[(size_t)route[m] * N + n0 + r] = acc[r][0][m] * gsc;
#endif
      }
    }
  }
}
"""

_MODS = {}


def _mod(MB, R, WARPS, GATED, K):
    import cupy as cp
    key = (MB, R, WARPS, GATED, K)
    if key not in _MODS:
        _MODS[key] = cp.RawModule(code=_CU_SRC, options=(
            "-std=c++17", f"-DMB={MB}", f"-DR={R}", f"-DWARPS={WARPS}", f"-DKMAX={K}", f"-DGATED={GATED}"))
    return _MODS[key]


def _p(t):
    return np.uint64(t.data_ptr())


SMALL_MAX = 32  # token counts up to this use the CUDA GEMV path; larger ones use Marlin (v9b)
# (MB, w13 (R, WARPS), w2 (R, WARPS)) per scored token count.
SMALL_CFG = {1: (1, (2, 8), (2, 8)), 8: (2, (2, 8), (2, 8)), 16: (4, (2, 8), (2, 8)), 32: (4, (2, 8), (2, 8))}
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
    MB, (R1, W1), (R2, W2) = cfg_override or _small_cfg(bs)
    NR = bs * top_k
    MAXM = NR
    MB = min(MB, NR)
    nwork = min(NR, NR // MB + min(count, NR))
    dev = x.device
    ib = torch.empty((NR + count + count * MAXM + nwork + 1,), device=dev, dtype=torch.int32)
    slot = ib[:NR]
    cnt = ib[NR:NR + count]
    lst = ib[NR + count:NR + count + count * MAXM]
    work = ib[NR + count + count * MAXM:NR + count + count * MAXM + nwork]
    nw = ib[NR + count + count * MAXM + nwork:]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    inter = torch.empty((NR, I), device=dev, dtype=torch.bfloat16)
    y32 = torch.empty((NR, H), device=dev, dtype=torch.float32)
    m1 = _mod(MB, R1, W1, 1, H)
    m2 = _mod(MB, R2, W2, 0, I)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((1,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                     np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(MB)))
        m1.get_function("moe_gemv")((I // (W1 * R1), nwork), (32 * W1,), (
            _p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(inter), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H), np.int32(top_k)))
        m2.get_function("moe_gemv")((H // (W2 * R2), nwork), (32 * W2,), (
            _p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(y32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I), np.int32(1)))
    out = torch.empty((bs, H), device=dev, dtype=x.dtype)
    _post_norm_sum_kernel[(bs,)](y32, state["norm_w"], data["topk_weights"].float().contiguous(), slot, out,
                                 POST_NORM_EPS, TOPK=top_k, H=H, num_warps=8)
    return out


def prepare(weights):
    start, count = LOCAL
    sl = slice(start, start + count)
    st = _prepare_marlin(weights)
    st.update({
        "w13": _permute_local(weights["w13"], start, count),
        "w13_sf": weights["w13_sf"][sl].contiguous().view(torch.uint8),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": _permute_local(weights["w2"], start, count),
        "w2_sf": weights["w2_sf"][sl].contiguous().view(torch.uint8),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    })
    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    for MB, (R1, W1), (R2, W2) in SMALL_CFG.values():  # compile the launch configs up front
        _mod(MB, R1, W1, 1, H)
        _mod(MB, R2, W2, 0, I)
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
BENCH_RW = [(2, 8), (1, 8), (4, 8), (2, 4), (4, 4), (1, 4), (8, 4)]
BENCH_MB = {1: (1,), 8: (1, 2, 4), 16: (2, 4), 32: (2, 4)}


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
        for MB in BENCH_MB[bs]:
            for rw in BENCH_RW:
                if time.time() - t0 > 300:
                    print("BENCH time cap", flush=True)
                    return
                try:
                    t, out = _time(lambda: _kernel_small(data, st, (MB, rw, rw)), flush)
                    err = float((out.float() - ref.float()).norm() / ref.float().norm())
                    rows.append((t, MB, rw, err))
                except Exception as ex:
                    print(f"BENCH bs={bs} MB={MB} rw={rw} error {type(ex).__name__}: {str(ex)[:160]}", flush=True)
        rows.sort()
        for t, MB, rw, err in rows:
            print(f"BENCH bs={bs} MB={MB} R,W={rw} {t:.1f}us err_vs_marlin={err:.4f}", flush=True)
        if rows:
            _, MB, best, _ = rows[0]
            for rw in BENCH_RW:
                t, _ = _time(lambda: _kernel_small(data, st, (MB, best, rw)), flush)
                print(f"BENCH bs={bs} MB={MB} w13={best} w2={rw} {t:.1f}us", flush=True)
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
