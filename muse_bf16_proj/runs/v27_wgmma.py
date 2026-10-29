# okc: problem=muse_bf16_proj
"""v27 (exploration): v17 plus an SM90a wgmma swap-AB GEMM (_WG_SRC; m64nNk16, N = tokens padded to 8 / 16 / 32,
FP32 accumulate) for qkv_gate, o, pooler and unpooler at 8 <= m <= 32 (v17's routes elsewhere). The weights are
repacked in prepare() (a pure permutation) into the no-swizzle K-major core-matrix image wgmma reads from shared
memory, streamed by 1-D cp.async.bulk (TMA) into 4 mbarrier stages; x is staged once per CTA in the same layout;
split-K partials are FP32 atomics, the last CTA of a 64-row tile writes BF16. Compiled in prepare() with NVRTC
(cuda.bindings, in-process, -arch=sm_90a) to a cubin loaded by CuPy.

v17: v14 with per-shape routing taken from judge numbers (the in-prepare bench misled): CUDA GEMV
only on the sliced unpooler (m <= 8); qkv_gate m <= 8 back on SGLang, pooler m <= 16 back on v7 Triton;
pre-transposed cuBLAS on qkv_gate m 16/32, pooler m 32, unpooler m >= 16, o m 128. BENCH off.
v14: v13 with per-shape routing from v13's bench (check cdcf4c435dce475a): the CUDA GEMV where it
won (qkv_gate m<=8, pooler m<=16, sliced unpooler), pre-transposed cuBLAS weights (torch.mm(x, W^T
contiguous)) where that won, v7 Triton for o; m bucketed (1, <=8, <=16, <=32). Bench adds RT (row tiles per warp).
v13: v7/v11 plus a hand-written CUDA tensor-core GEMV (NVRTC through CuPy, compiled in-process, no
subprocess) for the BF16 layers at m <= 32: W repacked (lossless permutation) into mma.sync m16n8k16
A-fragment order, one 16 B x load per lane per two mmas, split-K finished in-kernel (fp32 atomics plus a
last-CTA counter that writes BF16 and re-zeroes the workspace). Router unchanged (Triton hi + lo).
prepare() also benchmarks CUDA configs against SGLang's dispatch and pre-transposed cuBLAS (BENCH lines);
kernel_fn uses only the fixed CUDA_CFG table. A submit sets BENCH = False."""

import numpy as np
from typing import TypedDict

import torch
import triton
import triton.language as tl


_CU_SRC = r"""
// out[m, n] = sum_k x[m, k] * W[n, k] (bf16 in, fp32 accumulate, bf16 out), m <= 8*NB.
// Wp: W repacked per (16-row tile, 32-wide k step): 32 lanes x 16 bf16 (32 B) per lane, in mma.sync
// m16n8k16 A-fragment order with the k slots permuted so lane t owns k = 8t..8t+7 of the step
// (two mmas: k 8t..8t+3 then 8t+4..8t+7). The same permutation on x makes each B fragment one 16 B load.
// grid (NT / RT, S), block 32 * WARPS. Each (CTA split, warp) covers STEPS consecutive k32 steps.
// S > 1: fp32 atomics into ws (zero), last CTA of a row group (counter) writes bf16 and re-zeroes.
__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
extern "C" __global__ void __launch_bounds__(32 * WARPS)
gemv_mma(const uint4* __restrict__ Wp, const unsigned short* __restrict__ x, unsigned short* __restrict__ out,
         float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int STEPS, int S) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int K32 = K >> 5;
  const int chunk = blockIdx.y * WARPS + warp;
  const int ks0 = chunk * STEPS;
  float acc[RT][NB][4];
#pragma unroll
  for (int r = 0; r < RT; r++)
#pragma unroll
    for (int b = 0; b < NB; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[r][b][c] = 0.f;
  const uint4* xr[NB];
  bool xv[NB];
#pragma unroll
  for (int b = 0; b < NB; b++) {
    int tok = b * 8 + g;
    xv[b] = tok < M;
    xr[b] = reinterpret_cast<const uint4*>(x + (size_t)(xv[b] ? tok : 0) * K) + t;
  }
  const uint4* wr[RT];
#pragma unroll
  for (int r = 0; r < RT; r++)
    wr[r] = Wp + ((size_t)(blockIdx.x * RT + r) * K32 * 32 + lane) * 2;
  for (int s0 = 0; s0 < STEPS; s0 += U) {
    uint4 wa[U][RT][2], xb[U][NB];
#pragma unroll
    for (int u = 0; u < U; u++) {
      int ks = ks0 + s0 + u;
#pragma unroll
      for (int r = 0; r < RT; r++) {
        const uint4* p = wr[r] + (size_t)ks * 64;
        asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wa[u][r][0].x), "=r"(wa[u][r][0].y), "=r"(wa[u][r][0].z), "=r"(wa[u][r][0].w) : "l"(p));
        asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wa[u][r][1].x), "=r"(wa[u][r][1].y), "=r"(wa[u][r][1].z), "=r"(wa[u][r][1].w) : "l"(p + 1));
      }
#pragma unroll
      for (int b = 0; b < NB; b++) {
        if (xv[b]) xb[u][b] = __ldg(xr[b] + ks * 4); else xb[u][b] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < RT; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) {
          asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[r][b][0]), "+f"(acc[r][b][1]), "+f"(acc[r][b][2]), "+f"(acc[r][b][3])
            : "r"(wa[u][r][0].x), "r"(wa[u][r][0].y), "r"(wa[u][r][0].z), "r"(wa[u][r][0].w), "r"(xb[u][b].x), "r"(xb[u][b].y));
          asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[r][b][0]), "+f"(acc[r][b][1]), "+f"(acc[r][b][2]), "+f"(acc[r][b][3])
            : "r"(wa[u][r][1].x), "r"(wa[u][r][1].y), "r"(wa[u][r][1].z), "r"(wa[u][r][1].w), "r"(xb[u][b].z), "r"(xb[u][b].w));
        }
  }
  // CTA reduction: red[row 0..16*RT)[token 0..8*NB)
  __shared__ float red[16 * RT][8 * NB];
  for (int i = threadIdx.x; i < 16 * RT * 8 * NB; i += 32 * WARPS) (&red[0][0])[i] = 0.f;
  __syncthreads();
#pragma unroll
  for (int r = 0; r < RT; r++)
#pragma unroll
    for (int b = 0; b < NB; b++) {
      atomicAdd(&red[r * 16 + g][b * 8 + 2 * t], acc[r][b][0]);
      atomicAdd(&red[r * 16 + g][b * 8 + 2 * t + 1], acc[r][b][1]);
      atomicAdd(&red[r * 16 + g + 8][b * 8 + 2 * t], acc[r][b][2]);
      atomicAdd(&red[r * 16 + g + 8][b * 8 + 2 * t + 1], acc[r][b][3]);
    }
  __syncthreads();
  const int n0 = blockIdx.x * RT * 16;
  const int tot = 16 * RT * M;
  if (S == 1) {
    for (int i = threadIdx.x; i < tot; i += 32 * WARPS) {
      int row = i % (16 * RT), tok = i / (16 * RT);
      out[(size_t)tok * N + n0 + row] = f2bf(red[row][tok]);
    }
    return;
  }
  for (int i = threadIdx.x; i < tot; i += 32 * WARPS) {
    int row = i % (16 * RT), tok = i / (16 * RT);
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
    int row = i % (16 * RT), tok = i / (16 * RT);
    float* p = ws + (size_t)tok * N + n0 + row;
    float v = __ldcg(p);
    out[(size_t)tok * N + n0 + row] = f2bf(v);
    *p = 0.f;
  }
  if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
}
"""


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
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            state[name + "_p"] = _repack(weights[name])
            n = weights[name].shape[0]
            state[name + "_ws"] = torch.zeros((32, n), device=weights[name].device, dtype=torch.float32)
            state[name + "_t"] = weights[name].t().contiguous()
            state[name + "_cnt"] = torch.zeros((n // 16,), device=weights[name].device, dtype=torch.int32)
            state[name + "_wg"] = _wg_repack(weights[name])
            state[name + "_wgws"] = torch.zeros((WG_MAX_M, n), device=weights[name].device, dtype=torch.float32)
            state[name + "_wgcnt"] = torch.zeros((n // 64,), device=weights[name].device, dtype=torch.int32)
        for nt in (8, 16, 32):  # NVRTC-compile now; allow up to 160 KB of dynamic shared memory
            _wg_mod(nt).max_dynamic_shared_size_bytes = 160 * 1024
        torch.cuda.synchronize()
    if BENCH and weights["qkv_gate"].is_cuda:
        _bench(state)
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
    if layer != "router" and WG_MIN_M <= m <= WG_MAX_M and x.is_cuda:
        return _wg_gemm(x, layer, n, state)
    if layer != "router" and x.is_cuda:
        if m > TRITON_MAX_M:
            if layer in BIG_T or (layer == "o" and m <= 128):
                return torch.mm(x, state[layer + "_t"])
        else:
            c = CUDA_CFG.get((layer, layer == "unpooler" and n == 1024, _bucket(m)))
            if c == "T":
                return torch.mm(x, state[layer + "_t"][:, :n])
            if c is not None:
                return _cuda_gemv(x, layer, n, state, *c)
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


# (layer, unpooler sliced to 1024 rows, m bucket) -> (split S, warps per CTA, k32 steps per load batch U),
# "T" = pre-transposed cuBLAS, absent = v7 path (Triton CONFIGS / SGLang). Picked per shape from the
# judge numbers of v11 (ae407efff6954988), v14 (ee8f8ee6204d4b78), v16 (693c8d31f9134635).
CUDA_CFG = {
    ("qkv_gate", False, 16): "T", ("qkv_gate", False, 32): "T",
    ("pooler", False, 32): "T",
    ("unpooler", True, 1): (16, 4, 2), ("unpooler", True, 8): (4, 8, 4),
    ("unpooler", False, 16): "T", ("unpooler", False, 32): "T",
}
# Above 32 tokens: pre-transposed cuBLAS where v11's bench (check ae407efff6954988) had it ~1 us ahead.
BIG_T = {"pooler", "unpooler"}


def _bucket(m):
    return 1 if m == 1 else (8 if m <= 8 else (16 if m <= 16 else 32))


_MODS = {}

_WG_SRC = r"""
// Swap-AB BF16 GEMM for small m on SM90a tensor cores (wgmma, one warpgroup per CTA):
//   out[m, n] = sum_k x[m, k] * W[n, k], FP32 accumulate, BF16 out.
// D (64 weight rows x NT tokens) = A (W tile, 64 x 16, K-major, shared memory) * B (x^T tile, 16 x NT, K-major,
// shared memory), wgmma.mma_async m64nNTk16, NT = padded token count (8 / 16 / 32).
// Shared-memory operands use the no-swizzle ("interleave") canonical K-major layout: 8-row x 16-byte core
// matrices (128 contiguous bytes); LBO = byte stride between core matrices along K, SBO = along M / N.
// Wp (prepare-time repack, a pure permutation of W): per (64-row tile r, 64-k chunk c) one 8 KB image
//   [4 k16 steps u][2 k-halves kg][8 row groups rg][8 rows][8 k] -> u*2048 + kg*1024 + rg*128 + row*16 + e*2 bytes,
// so one k16 step's A operand starts at u*2048 with LBO 1024, SBO 128. Images of a tile are contiguous over c,
// so a CTA's k range is one contiguous run, streamed with 1-D cp.async.bulk (TMA) into ST stages (mbarrier).
// x (B) is staged once per CTA for its k range by the threads: per chunk c, step u: [kg][token][8 k] ->
//   c*(NT*128) + u*(NT*32) + kg*(NT*16) + tok*16 bytes (LBO NT*16, SBO 128); tokens >= M are zero.
// grid (N / 64, S); block 128. S > 1: FP32 atomics into ws (zeroed), the last CTA of a tile (counter) writes
// BF16 and re-zeroes its rows of ws and its counter.
typedef unsigned long long u64;
__device__ __forceinline__ unsigned su32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ u64 desc(unsigned addr, unsigned lbo, unsigned sbo) {
  return (u64)((addr & 0x3FFFF) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32);  // swizzle 0, base 0
}
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
#define NACC (NT / 2)
__device__ __forceinline__ void wg_mma(float* d, u64 da, u64 db) {
#if NT == 8
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db));
#elif NT == 16
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 0, 0;\n}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(da), "l"(db));
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, "
      "%12, %13, %14, %15}, %16, %17, p, 1, 1, 0, 0;\n}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db));
#endif
}
// dynamic smem: [ST stages x 8 KB of W][NCH chunks x NT*128 B of x]
extern "C" __global__ void __launch_bounds__(128)
gemm_wg(const uint4* __restrict__ Wp, const unsigned short* __restrict__ x, unsigned short* __restrict__ out,
        float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int NCH, int S) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) u64 bar[ST];
  __shared__ int last;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int r = blockIdx.x, s = blockIdx.y;
  const int c0 = s * NCH;                       // first 64-k chunk of this CTA
  unsigned char* sw = smem;
  unsigned char* sx = smem + ST * 8192;
  const unsigned char* src = reinterpret_cast<const unsigned char*>(Wp) + ((size_t)r * (K / 64) + c0) * 8192;
  if (tid == 0) {
    for (int i = 0; i < ST; i++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(&bar[i])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0)
    for (int j = 0; j < ST && j < NCH; j++) issue(su32(sw + j * 8192), src + (size_t)j * 8192, 8192, su32(&bar[j]));
  // stage x for k in [c0*64, (c0+NCH)*64): 16-byte pieces (token, 8 k)
  const int pieces = NT * NCH * 8;
  for (int i = tid; i < pieces; i += 128) {
    int tok = i % NT, j = i / NT;               // j: 8-k group within the CTA's range
    int c = j >> 3, u = (j >> 1) & 3, kg = j & 1;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tok < M) v = *reinterpret_cast<const uint4*>(x + (size_t)tok * K + (size_t)c0 * 64 + j * 8);
    *reinterpret_cast<uint4*>(sx + c * (NT * 128) + u * (NT * 32) + kg * (NT * 16) + tok * 16) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic-proxy writes -> visible to wgmma
  __syncthreads();
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; i++) acc[i] = 0.f;
  const unsigned swa = su32(sw), sxa = su32(sx);
  for (int c = 0; c < NCH; c++) {
    const int slot = c % ST;
    mb_wait(su32(&bar[slot]), (c / ST) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int u = 0; u < 4; u++)
      wg_mma(acc, desc(swa + slot * 8192 + u * 2048, 1024, 128),
             desc(sxa + c * (NT * 128) + u * (NT * 32), NT * 16, 128));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    __syncthreads();                              // every warp is done reading this stage
    if (tid == 0 && c + ST < NCH) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue(su32(sw + slot * 8192), src + (size_t)(c + ST) * 8192, 8192, su32(&bar[slot]));
    }
  }
  // D fragment: warp w rows 16w + g (+8); n8 block j: acc[4j + 0/1] = (row, col 8j + 2t + 0/1), [4j + 2/3] row + 8
  const int n0 = r * 64 + warp * 16 + g;
  if (S == 1) {
#pragma unroll
    for (int i = 0; i < NACC; i++) {
      int j = i >> 2, q = i & 3;
      int tok = 8 * j + 2 * t + (q & 1), row = n0 + (q >> 1) * 8;
      if (tok < M) out[(size_t)tok * N + row] = f2bf(acc[i]);
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    int j = i >> 2, q = i & 3;
    int tok = 8 * j + 2 * t + (q & 1), row = n0 + (q >> 1) * 8;
    if (tok < M) atomicAdd(ws + (size_t)tok * N + row, acc[i]);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = (atomicAdd(cnt + r, 1) == S - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int i = tid; i < 64 * M; i += 128) {
    int row = r * 64 + (i % 64), tok = i / 64;
    float* p = ws + (size_t)tok * N + row;
    out[(size_t)tok * N + row] = f2bf(__ldcg(p));
    *p = 0.f;
  }
  if (tid == 0) cnt[r] = 0;
}
"""
_WG = {}
# layer -> split S (64-k chunks per CTA = K / 64 / S): ~512 CTAs of one warpgroup each.
WG_SPLIT = {"qkv_gate": 16, "o": 4, "pooler": 8, "unpooler": 4, "unpooler_s": 16}
WG_MIN_M, WG_MAX_M, WG_ST = 8, 32, 4


def _wg_mod(NT):
    """NVRTC (cuda.bindings, in-process) -> cubin for sm_90a (wgmma needs the arch-specific target) -> CuPy."""
    if NT not in _WG:
        import os
        import tempfile
        import cupy as cp
        from cuda.bindings import nvrtc
        err, prog = nvrtc.nvrtcCreateProgram(_WG_SRC.encode(), b"wg.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        opts = [b"-arch=sm_90a", b"-std=c++17", f"-DNT={NT}".encode(), f"-DST={WG_ST}".encode()]
        (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="ignore"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        fd, path = tempfile.mkstemp(suffix=".cubin")
        with os.fdopen(fd, "wb") as f:
            f.write(cubin)
        _WG[NT] = cp.RawModule(path=path).get_function("gemm_wg")
        os.unlink(path)
    return _WG[NT]


def _wg_repack(w):
    """[N, K] bf16 -> per (64-row tile r, 64-k chunk c) an 8 KB image [u 4][kg 2][rg 8][row 8][e 8] (a permutation)."""
    n, k = w.shape
    v = w.contiguous().view(n // 64, 8, 8, k // 64, 4, 2, 8)
    return v.permute(0, 3, 4, 5, 1, 2, 6).contiguous().view(-1)


def _wg_gemm(x, layer, n, state):
    import cupy as cp
    m, k = x.shape
    x = x.contiguous()
    assert x.data_ptr() % 16 == 0
    nt = 8 if m <= 8 else (16 if m <= 16 else 32)
    S = WG_SPLIT["unpooler_s" if (layer == "unpooler" and n == 1024) else layer]
    nch = k // 64 // S
    f = _wg_mod(nt)
    smem = WG_ST * 8192 + nch * nt * 128
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        f((n // 64, S), (128,), (
            np.uint64(state[layer + "_wg"].data_ptr()), np.uint64(x.data_ptr()), np.uint64(out.data_ptr()),
            np.uint64(state[layer + "_wgws"].data_ptr()), np.uint64(state[layer + "_wgcnt"].data_ptr()),
            np.int32(m), np.int32(n), np.int32(k), np.int32(nch), np.int32(S)), shared_mem=smem)
    return out


def _mod(NB, WARPS, U, RT=1):
    import cupy as cp
    key = (NB, WARPS, U, RT)
    if key not in _MODS:
        _MODS[key] = cp.RawModule(code=_CU_SRC, options=(
            "-std=c++17", f"-DNB={NB}", f"-DWARPS={WARPS}", f"-DU={U}", f"-DRT={RT}"))
    return _MODS[key]


def _repack(w):
    """[N, K] bf16 -> same values permuted: (N/16, K/32, lane g 8, lane t 4, mma 2, a-half ch 2, row-half 2, 2)."""
    n, k = w.shape
    v = w.contiguous().view(n // 16, 2, 8, k // 32, 4, 2, 2, 2)
    return v.permute(0, 3, 2, 4, 5, 6, 1, 7).contiguous().view(-1)


def _cuda_gemv(x, layer, n, state, S, WARPS, U, RT=1):
    import cupy as cp
    m, k = x.shape
    x = x.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    nb = (m + 7) // 8
    steps = (k // 32) // (S * WARPS)
    f = _mod(nb, WARPS, U, RT).get_function("gemv_mma")
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        f((n // 16 // RT, S), (32 * WARPS,), (
            np.uint64(state[layer + "_p"].data_ptr()), np.uint64(x.data_ptr()), np.uint64(out.data_ptr()),
            np.uint64(state[layer + "_ws"].data_ptr()), np.uint64(state[layer + "_cnt"].data_ptr()),
            np.int32(m), np.int32(n), np.int32(k), np.int32(steps), np.int32(S)))
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


# ------------------------------- exploration bench (check runs only) -------------------------------
BENCH = False
_BENCHED = []


def _time(fn, flush, reps=40):
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


def _tri(x, w, n, cfg):
    bn, bk, split, warps, stages = cfg
    m, k = x.shape
    bm = _bm(m)
    if split > 1:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _swapab_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm), split)](
        x, w, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0), BM=bm, BN=bn, BK=bk, SPLIT_K=split,
        HAS_LO=False, OUT_BF16=(split == 1), num_warps=warps, num_stages=stages)
    return out.to(x.dtype) if split > 1 else out


def _bench(state):
    if _BENCHED:
        return
    _BENCHED.append(1)
    import time
    t0 = time.time()
    dev = state["qkv_gate"].device
    flush = torch.empty((512 * 1024 * 1024 // 4,), device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(5)
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl as disp
    tri_cfgs = [(64, 128, 1, 4, 4), (32, 256, 1, 4, 4), (16, 256, 4, 4, 4), (32, 256, 2, 4, 4), (64, 256, 2, 8, 3)]
    for layer, n_full in (("qkv_gate", 2304), ("o", 8192), ("pooler", 4096), ("unpooler", 8192)):
        W = state[layer]
        for m in (1, 8, 16, 32):
            n = 1024 if (layer == "unpooler" and m <= 8) else n_full
            Wn = W[:n]
            Wt = Wn.t().contiguous()
            x = torch.randn((m, W.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            tb, ref = _time(lambda: disp(x, Wn, None), flush)
            res = []
            cands = {"mm_Wt": lambda: torch.mm(x, Wt)}
            if layer == "o":
                cands["tri_o"] = lambda: _tri(x, Wn, n, CONFIGS["o"])
            K32 = W.shape[1] // 32
            for S in (1, 2, 4, 8, 16):
                for WARPS in (2, 4, 8):
                    for U in (2, 4, 8):
                      for RT in ((1, 2, 4) if m >= 16 else (1, 2)):
                        if K32 % (S * WARPS) or (K32 // (S * WARPS)) % U or (n // 16) % RT:
                            continue
                        if (n // 16 // RT) * S < 66:
                            continue
                        cands[f"cu{S},{WARPS},{U},{RT}"] = (lambda S=S, WARPS=WARPS, U=U, RT=RT: _cuda_gemv(x, layer, n, state, S, WARPS, U, RT))
            for name, fn in cands.items():
                try:
                    t, out = _time(fn, flush)
                    err = float((out.float() - ref.float()).norm() / ref.float().norm())
                    res.append((t, name, err))
                except Exception as ex:
                    res.append((1e9, name + " ERR " + type(ex).__name__ + ":" + str(ex)[:60], 0.0))
            res.sort()
            line = " ".join(f"{nm}={t:.2f}({e:.0e})" for t, nm, e in res[:8])
            print(f"BENCH {layer} m={m} n={n} sglang={tb:.2f}us cfg={_time(lambda: kernel_fn({'x': x, 'config': {'m': m, 'n': n, 'k': W.shape[1], 'layer': layer}}, state), flush)[0]:.2f}us | {line}", flush=True)
    print(f"BENCH done {time.time() - t0:.0f}s", flush=True)
