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
