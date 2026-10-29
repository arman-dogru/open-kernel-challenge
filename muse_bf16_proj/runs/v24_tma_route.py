# okc: problem=muse_bf16_proj
"""v24: v23 routed from its judge table (check 42f8408fae1a4a8b) plus a 4-warp config for m >= 2:
- m == 1: v23's 1-warp, 8/16-way split config on qkv_gate (1.072x) and the sliced unpooler (1.084x);
  pooler m 1 back on v7 Triton (1.085x vs 1.058x).
- m 2-32 (exploration): 4 warps per CTA split K inside the CTA and reduce in shared memory, so the
  global FP32 atomics (what sank v23 at m >= 8: 0.46-0.91x) drop 4-16x; fewer CTA splits.

v23: v7 plus a hand-written SM90 kernel for the BF16 layers at m <= 32 (qkv_gate, pooler,
unpooler; o stays on v7's Triton, router on v7's Triton hi + lo): weights repacked (lossless permutation,
as v13/v17) into mma.sync m16n8k16 A-fragment order, so each warp's slice of a 16-row tile is one
contiguous run of bytes; it is streamed global -> shared by the Hopper TMA bulk-copy engine
(cp.async.bulk, completion on an mbarrier, ST stages of SS KB in flight per warp) instead of per-thread
loads (v13-v18 topped out near 3.2 TB/s with register-held loads). x is read with one 16 B load per lane
per 32-wide k step; split-K over CTAs, finished in-kernel (FP32 atomics + last-CTA counter that writes
BF16 and re-zeroes its workspace). NVRTC through CuPy, compiled in prepare() (get_function on every
config). Exploration route: every m <= 32 shape of those three layers takes the new kernel, so this
check's per-shape table decides the final routing."""

from typing import TypedDict

import numpy as np
import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


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


# (BN, BK, SPLIT_K, num_warps, num_stages) per layer, for m <= 32, picked per shape from H200
# check runs (v1/v2 in runs/). Layers set to None keep SGLang's cuBLAS path, which won there.
CONFIGS = {
    "qkv_gate": None,
    "o": (64, 128, 1, 4, 4),
    "pooler": (32, 256, 1, 4, 4),
    "unpooler_s": None,
    "unpooler": None,
    "router": (16, 256, 8, 4, 4),
}
# Above this m, cuBLAS (SGLang's path) wins on every layer so far.
TRITON_MAX_M = 32


_CU_SRC = r"""
// out[m, n] = sum_k x[m, k] * W[n, k], bf16 in, fp32 acc, bf16 out, m <= 8*NB. SM90.
// Wp: v13/v17 repack: [N/16 rowtiles][K/32 steps][32 lanes][2 x uint4] (mma.sync m16n8k16 A fragments,
// k permuted so lane t owns k = 8t..8t+7 of a step; x then needs one 16 B load per lane per step).
// Each warp streams its rowtile's k range (contiguous in Wp) into shared memory with cp.async.bulk
// (TMA bulk copy, 1-D) in stages of SS steps (SS KB), ST stages in flight, completion on an mbarrier.
// grid (N/16, S), block 32*WARPS; warp w of CTA (rt, s) covers steps [(s*WARPS+w)*STEPS, +STEPS).
// S > 1: fp32 atomics into ws (zeroed), the last CTA of a rowtile (counter) writes bf16 and re-zeroes.
__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ unsigned su32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_wait(unsigned a, unsigned ph) {
  unsigned ok = 0;
  while (!ok) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(a), "r"(ph) : "memory");
  }
}
__device__ __forceinline__ void issue(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
extern "C" __global__ void __launch_bounds__(32 * WARPS)
gemv_tma(const uint4* __restrict__ Wp, const unsigned short* __restrict__ x, unsigned short* __restrict__ out,
         float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int STEPS, int S) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) unsigned long long bar[WARPS][ST];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int K32 = K >> 5;
  const int ks0 = (blockIdx.y * WARPS + warp) * STEPS;
  const int nst = STEPS / SS;
  uint4* mys = sm + warp * (ST * SS * 64);
  const uint4* src = Wp + ((size_t)blockIdx.x * K32 + ks0) * 64;
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < ST; s++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(&bar[warp][s])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncwarp();
  if (lane == 0) {
    for (int j = 0; j < ST && j < nst; j++)
      issue(su32(mys + j * SS * 64), src + (size_t)j * SS * 64, SS * 1024, su32(&bar[warp][j]));
  }
  float acc[NB][4];
#pragma unroll
  for (int b = 0; b < NB; b++)
#pragma unroll
    for (int c = 0; c < 4; c++) acc[b][c] = 0.f;
  const uint4* xr[NB];
  bool xv[NB];
#pragma unroll
  for (int b = 0; b < NB; b++) {
    int tok = b * 8 + g;
    xv[b] = tok < M;
    xr[b] = reinterpret_cast<const uint4*>(x + (size_t)(xv[b] ? tok : 0) * K) + t;
  }
  for (int j = 0; j < nst; j++) {
    const int slot = j % ST;
    uint4 xb[SS][NB];
#pragma unroll
    for (int u = 0; u < SS; u++)
#pragma unroll
      for (int b = 0; b < NB; b++) {
        int ks = ks0 + j * SS + u;
        if (xv[b]) xb[u][b] = __ldg(xr[b] + ks * 4); else xb[u][b] = make_uint4(0, 0, 0, 0);
      }
    mb_wait(su32(&bar[warp][slot]), (j / ST) & 1);
    const uint4* ws_ = mys + slot * SS * 64;
#pragma unroll
    for (int u = 0; u < SS; u++) {
      uint4 a0 = ws_[u * 64 + lane * 2], a1 = ws_[u * 64 + lane * 2 + 1];
#pragma unroll
      for (int b = 0; b < NB; b++) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[b][0]), "+f"(acc[b][1]), "+f"(acc[b][2]), "+f"(acc[b][3])
            : "r"(a0.x), "r"(a0.y), "r"(a0.z), "r"(a0.w), "r"(xb[u][b].x), "r"(xb[u][b].y));
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[b][0]), "+f"(acc[b][1]), "+f"(acc[b][2]), "+f"(acc[b][3])
            : "r"(a1.x), "r"(a1.y), "r"(a1.z), "r"(a1.w), "r"(xb[u][b].z), "r"(xb[u][b].w));
      }
    }
    __syncwarp();
    if (lane == 0 && j + ST < nst) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue(su32(mys + slot * SS * 64), src + (size_t)(j + ST) * SS * 64, SS * 1024, su32(&bar[warp][slot]));
    }
  }
  __shared__ float red[16][8 * NB];
  for (int i = threadIdx.x; i < 16 * 8 * NB; i += 32 * WARPS) (&red[0][0])[i] = 0.f;
  __syncthreads();
#pragma unroll
  for (int b = 0; b < NB; b++) {
    atomicAdd(&red[g][b * 8 + 2 * t], acc[b][0]);
    atomicAdd(&red[g][b * 8 + 2 * t + 1], acc[b][1]);
    atomicAdd(&red[g + 8][b * 8 + 2 * t], acc[b][2]);
    atomicAdd(&red[g + 8][b * 8 + 2 * t + 1], acc[b][3]);
  }
  __syncthreads();
  const int n0 = blockIdx.x * 16;
  const int tot = 16 * M;
  if (S == 1) {
    for (int i = threadIdx.x; i < tot; i += 32 * WARPS) {
      int row = i % 16, tok = i / 16;
      out[(size_t)tok * N + n0 + row] = f2bf(red[row][tok]);
    }
    return;
  }
  for (int i = threadIdx.x; i < tot; i += 32 * WARPS) {
    int row = i % 16, tok = i / 16;
    atomicAdd(ws + (size_t)tok * N + n0 + row, red[row][tok]);
  }
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = (atomicAdd(cnt + blockIdx.x, 1) == S - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int i = threadIdx.x; i < tot; i += 32 * WARPS) {
    int row = i % 16, tok = i / 16;
    float* p = ws + (size_t)tok * N + n0 + row;
    float v = __ldcg(p);
    out[(size_t)tok * N + n0 + row] = f2bf(v);
    *p = 0.f;
  }
  if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
}
"""

# layer -> (S splits over CTAs, WARPS per CTA, SS k32 steps per stage, ST stages in flight)
# (layer, m == 1) -> (S splits over CTAs, WARPS per CTA, SS k32 steps per stage, ST stages); absent = v7 path
TMA_CFG = {("qkv_gate", True): (8, 1, 4, 4), ("unpooler_s", True): (16, 1, 4, 4),
           ("qkv_gate", False): (2, 4, 4, 4), ("pooler", False): (1, 4, 4, 4),
           ("unpooler", False): (1, 4, 4, 4), ("unpooler_s", False): (4, 4, 4, 4)}
TMA_MAX_M = 32
_MODS = {}


def _mod(NB, WARPS, SS, ST):
    import cupy as cp
    key = (NB, WARPS, SS, ST)
    if key not in _MODS:
        _MODS[key] = cp.RawModule(code=_CU_SRC, options=(
            "-std=c++17", f"-DNB={NB}", f"-DWARPS={WARPS}", f"-DSS={SS}", f"-DST={ST}"))
    return _MODS[key]


def _repack(w):
    """[N, K] bf16 -> the same values permuted (v13/v17 layout):
    (N/16, K/32, lane g 8, lane t 4, mma 2, a-half 2, row-half 2, 2)."""
    n, k = w.shape
    v = w.contiguous().view(n // 16, 2, 8, k // 32, 4, 2, 2, 2)
    return v.permute(0, 3, 2, 4, 5, 6, 1, 7).contiguous().view(-1)


def _key(layer, n):
    return "unpooler_s" if (layer == "unpooler" and n == 1024) else layer


def _tma_route(layer, n, m):
    return m <= TMA_MAX_M and (_key(layer, n), m == 1) in TMA_CFG


def _tma_gemv(x, layer, n, state):
    import cupy as cp
    m, k = x.shape
    x = x.contiguous()
    assert x.data_ptr() % 16 == 0
    S, WARPS, SS, ST = TMA_CFG[(_key(layer, n), m == 1)]
    steps = (k // 32) // (S * WARPS)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    f = _mod((m + 7) // 8 if m <= 16 else 4, WARPS, SS, ST).get_function("gemv_tma")
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        f((n // 16, S), (32 * WARPS,), (
            np.uint64(state[layer + "_p"].data_ptr()), np.uint64(x.data_ptr()), np.uint64(out.data_ptr()),
            np.uint64(state[layer + "_ws"].data_ptr()), np.uint64(state[layer + "_cnt"].data_ptr()),
            np.int32(m), np.int32(n), np.int32(k), np.int32(steps), np.int32(S)),
          shared_mem=WARPS * ST * SS * 1024)
    return out


def _bm(m):
    return 16 if m <= 16 else (32 if m <= 32 else 64)


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    if weights["qkv_gate"].is_cuda:
        for name in ("qkv_gate", "pooler", "unpooler"):
            n = weights[name].shape[0]
            state[name + "_p"] = _repack(weights[name])
            state[name + "_ws"] = torch.zeros((TMA_MAX_M, n), device=weights[name].device, dtype=torch.float32)
            state[name + "_cnt"] = torch.zeros((n // 16,), device=weights[name].device, dtype=torch.int32)
        for cfg in set(TMA_CFG.values()):  # compile now (CuPy is lazy); allow > 48 KB dynamic smem
            for nb in (1, 2, 4):
                f = _mod(nb, cfg[1], cfg[2], cfg[3]).get_function("gemv_tma")
                f.max_dynamic_shared_size_bytes = cfg[1] * cfg[3] * cfg[2] * 1024
        torch.cuda.synchronize()
    return state


def _baseline(x, layer, n, state):
    """SGLang's own path for the BF16 layers (as in baseline.py), kept where it is faster.
    (The router never comes here: Triton up to 32 tokens, _router_stacked above.)"""
    if x.device.type == "cpu":  # local CPU testing only (no SGLang / cuBLAS bf16 path)
        return x @ state[layer][:n].T
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m, k = x.shape
    if layer == "router" and m > TRITON_MAX_M:
        return _router_stacked(x, state)
    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
    if x.is_cuda and _tma_route(layer, n, m):
        return _tma_gemv(x, layer, n, state)
    key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
    if CONFIGS[key] is None:
        return _baseline(x, layer, n, state)
    bn, bk, split, warps, stages = CONFIGS[key]
    bm = _bm(m)
    if layer == "router":
        w, lo = state["router_hi"], state["router_lo"]
    else:
        w, lo = state[layer], state[layer]
    x = x.contiguous()
    if split > 1:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm), split)
    _swapab_gemm[grid](x, w, lo, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                       BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_LO=(layer == "router"),
                       OUT_BF16=(split == 1), num_warps=warps, num_stages=stages)
    if layer != "router" and split > 1:
        out = out.to(x.dtype)
    return out


def _router_stacked(x, state):
    """Router above 32 tokens: SGLang's hi + lo pair as ONE cuBLAS GEMM against the stacked
    [hi; lo] weights (FP32 out), then hi-part + lo-part: the same precision as SGLang's two
    torch.mm calls (BF16 inputs, FP32 accumulation, FP32 add), in one GEMM launch
    (1.56-1.60x at 128 / 256 tokens in check c3f668686d154849)."""
    if x.device.type == "cpu":  # local CPU testing only
        both = x.float() @ state["router_hilo_t"].float()
    else:
        both = torch.mm(x, state["router_hilo_t"], out_dtype=torch.float32)
    n = both.shape[1] // 2
    return both[:, :n] + both[:, n:]
