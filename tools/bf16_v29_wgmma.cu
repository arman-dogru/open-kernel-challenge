// v28: swap-AB BF16 GEMM on SM90a (wgmma, one warpgroup per CTA), fully software-pipelined.
//   out[m, n] = sum_k x[m, k] * W[n, k], FP32 accumulate, BF16 out.
// Each CTA owns a 64-row weight tile and a short k range of NCH 64-k chunks (NCH == ST, chosen by the split S so
// NCH is small). It gives every chunk its own W stage (NO slot reuse), issues all NCH TMA loads up front, then
// runs the wgmma over the chunks and waits once at the end: every TMA is in flight at once and all of the compute
// overlaps the loads. Layout and descriptors are v27's (no-swizzle K-major core matrices), proven on the judge.
// grid (N / 64, S); block 128. S > 1: each split stores FP32 partials to ws[s][tok][n] ([S][32][N]); the last CTA of a
// tile (counter) sums them in a fixed order (deterministic) and writes BF16. No atomics on data, no re-zeroing.
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
// dynamic smem: [ST stages x 8 KB W][ST chunks x NT*128 B x]
extern "C" __global__ void __launch_bounds__(128)
gemm_wg(const uint4* __restrict__ Wp, const unsigned short* __restrict__ x, unsigned short* __restrict__ out,
        float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int NCH, int S) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) u64 bar[ST];
  __shared__ int last;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int r = blockIdx.x, s = blockIdx.y;
  const int c0 = s * ST;                        // first 64-k chunk of this CTA (NCH == ST)
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
#pragma unroll
    for (int j = 0; j < ST; j++) issue(su32(sw + j * 8192), src + (size_t)j * 8192, 8192, su32(&bar[j]));
  // stage x for this CTA's k range: 16-byte pieces (token, 8 k) into [chunk][u 4][kg 2][token][8 k]
  const int pieces = NT * ST * 8;
  for (int i = tid; i < pieces; i += 128) {
    int tok = i % NT, j = i / NT;               // j: 8-k group within the range (ST chunks x 8 groups)
    int c = j >> 3, u = (j >> 1) & 3, kg = j & 1;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tok < M) v = *reinterpret_cast<const uint4*>(x + (size_t)tok * K + (size_t)c0 * 64 + j * 8);
    *reinterpret_cast<uint4*>(sx + c * (NT * 128) + u * (NT * 32) + kg * (NT * 16) + tok * 16) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic-proxy x writes -> visible to wgmma
  __syncthreads();
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; i++) acc[i] = 0.f;
  const unsigned swa = su32(sw), sxa = su32(sx);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  for (int c = 0; c < ST; c++) {
    mb_wait(su32(&bar[c]), 0);                   // chunk c's W (loaded once; parity 0)
#pragma unroll
    for (int u = 0; u < 4; u++)
      wg_mma(acc, desc(swa + c * 8192 + u * 2048, 1024, 128),
             desc(sxa + c * (NT * 128) + u * (NT * 32), NT * 16, 128));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");  // all groups retired; acc final
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
  // v29: plain partial stores ws[s][tok][n] (no atomics, no zeroing); the last CTA of the tile sums s = 0..S-1 in order.
  float* wsp = ws + (size_t)s * 32 * N;
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    int j = i >> 2, q = i & 3;
    int tok = 8 * j + 2 * t + (q & 1), row = n0 + (q >> 1) * 8;
    if (tok < M) __stcg(wsp + (size_t)tok * N + row, acc[i]);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = (atomicAdd(cnt + r, 1) == S - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  // float4 over 4 adjacent rows: 16 * M items, S independent 16-byte loads each (4 accumulators for ILP).
  const size_t st4 = (size_t)32 * N / 4;
  for (int i = tid; i < 16 * M; i += 128) {
    int tok = i >> 4, row = r * 64 + (i & 15) * 4;
    const float4* p = reinterpret_cast<const float4*>(ws + (size_t)tok * N + row);
    float4 a[4];
#pragma unroll
    for (int q = 0; q < 4; q++) a[q] = make_float4(0.f, 0.f, 0.f, 0.f);
    int k = 0;
    for (; k + 4 <= S; k += 4) {  // fixed order -> deterministic
#pragma unroll
      for (int q = 0; q < 4; q++) {
        float4 v = __ldcg(p + (size_t)(k + q) * st4);
        a[q].x += v.x; a[q].y += v.y; a[q].z += v.z; a[q].w += v.w;
      }
    }
    for (; k < S; k++) {
      float4 v = __ldcg(p + (size_t)k * st4);
      a[0].x += v.x; a[0].y += v.y; a[0].z += v.z; a[0].w += v.w;
    }
    uint2 pk;  // 4 BF16 = 8 bytes, 8-byte aligned (row % 4 == 0)
    pk.x = f2bf((a[0].x + a[1].x) + (a[2].x + a[3].x)) | ((unsigned)f2bf((a[0].y + a[1].y) + (a[2].y + a[3].y)) << 16);
    pk.y = f2bf((a[0].z + a[1].z) + (a[2].z + a[3].z)) | ((unsigned)f2bf((a[0].w + a[1].w) + (a[2].w + a[3].w)) << 16);
    *reinterpret_cast<uint2*>(out + (size_t)tok * N + row) = pk;
  }
  if (tid == 0) cnt[r] = 0;
}
