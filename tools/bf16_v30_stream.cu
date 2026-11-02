// bf16 v30: weight-streaming skinny GEMM out[m, n] = x[m, K] . W[n, K]^T for m <= 32 (BF16 in, FP32 accumulate,
// BF16 out), sm_90a, mma.sync m16n8k16 swap-AB (W rows = the MMA's M side, tokens = its N side).
// - W is repacked once in prepare() into A-fragment order: [rowtile (16 rows)][pair (32 k)][step s][lane][4 x b32],
//   so a warp streams its (rowtile, k-range) as consecutive 512-byte LDG.128s (L1 no-allocate, L2 evict-first) and
//   each 16-byte load is one MMA's whole A fragment. k is permuted inside each 32-wide pair (lane t, step s holds
//   k = 8t + 4s + 0..3) identically for W and x, so x's B fragments are one 16-byte load per (token tile, pair).
// - CTA = RG x WK warps: RG rowtile groups (RT rowtiles per warp) x WK warps along k; CS CTAs per cluster split k
//   further. Each warp loads GP pairs (GP * RT KB) per chunk; PFD chunks ahead are bulk-prefetched into L2
//   (cp.async.bulk.prefetch.L2), which is what keeps enough bytes in flight (ptxas serializes the register loads).
// - Split-K reduction without atomics: in-CTA through shared memory (fixed order), then across the cluster through
//   distributed shared memory into rank 0, which writes BF16. Deterministic; no workspace.
// - Persistent: grid = min(units, cap) clusters; a cluster loops over units (unit = RG * RT rowtiles).
typedef unsigned int u32;
typedef unsigned short u16;

__device__ __forceinline__ uint4 ldw(const uint4* p, unsigned long long pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void pf_l2(const void* p, u32 bytes) {  // bulk L2 prefetch (fire and forget)
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint4 ldx(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even (finite)
  u32 u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (u16)(u >> 16);
}
__device__ __forceinline__ u32 cluster_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_dsmem(const float* p, u32 rank) {
  u32 a = (u32)__cvta_generic_to_shared(p), ra;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}

template <int RT, int NT, int WK, int RG, int CS, int GP, int PFD = 2>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(32 * WK * RG)
bf16_stream(const uint4* __restrict__ W, const u16* __restrict__ x, int ldx_, u16* __restrict__ out, int m, int n,
            int K, int units) {
  constexpr int E = RG * RT * NT * 128;  // FP32 partial outputs per CTA per unit
  extern __shared__ float red[];         // [WK][E]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rg = warp / WK, ks = warp % WK;
  const int g = lane >> 2, t = lane & 3;
  const u32 rank = CS > 1 ? cluster_rank() : 0u;
  const int cid = blockIdx.x / CS, ncl = gridDim.x / CS;
  const int KP = K >> 5;                 // pairs per rowtile
  const int PW = KP / (WK * CS);         // pairs per warp
  const int p0 = ((int)rank * WK + ks) * PW;
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  for (int u = cid; u < units; u += ncl) {
    const int rt0 = (u * RG + rg) * RT;
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][j][c] = 0.f;
    const uint4* wp[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wp[r] = W + ((size_t)(rt0 + r) * KP + p0) * 64 + lane;
    // ptxas interleaves the A loads with their MMAs (~2 in flight per warp), so the DRAM latency is hidden by
    // bulk L2 prefetches of the warp's next chunks instead: PFD chunks (GP KB per rowtile each) ahead.
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
        for (int q = 0; q < PFD && q * GP < PW; ++q) pf_l2(wp[r] - lane + (size_t)q * GP * 64, GP * 1024);
    }
    for (int c = 0; c < PW; c += GP) {
      if (lane == 0 && c + PFD * GP < PW) {
#pragma unroll
        for (int r = 0; r < RT; ++r) pf_l2(wp[r] - lane + (size_t)(c + PFD * GP) * 64, GP * 1024);
      }
      uint4 a[GP][RT][2];
#pragma unroll
      for (int i = 0; i < GP; ++i)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          a[i][r][0] = ldw(wp[r] + (size_t)(c + i) * 64, pol);
          a[i][r][1] = ldw(wp[r] + (size_t)(c + i) * 64 + 32, pol);
        }
#pragma unroll
      for (int i = 0; i < GP; ++i) {
        const int kb = ((p0 + c + i) << 5) + 8 * t;
        uint4 b[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int col = 8 * j + g;
          b[j] = col < m ? ldx(reinterpret_cast<const uint4*>(x + (size_t)col * ldx_ + kb)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            mma(acc[r][j], a[i][r][0].x, a[i][r][0].y, a[i][r][0].z, a[i][r][0].w, b[j].x, b[j].y);
            mma(acc[r][j], a[i][r][1].x, a[i][r][1].y, a[i][r][1].z, a[i][r][1].w, b[j].z, b[j].w);
          }
      }
    }
    // in-CTA split-K reduction (fixed order over ks)
    float* my = red + ks * E + rg * RT * NT * 128;
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) my[((r * NT + j) * 4 + c) * 32 + lane] = acc[r][j][c];
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < WK; ++w) s += red[w * E + e];
      red[e] = s;
    }
    if (CS > 1) cluster_sync(); else __syncthreads();
    if (rank == 0) {
      for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float s = red[e];
#pragma unroll
        for (int q = 1; q < CS; ++q) s += ld_dsmem(red + e, (u32)q);
        const int ln = e & 31, c = (e >> 5) & 3, rj = e >> 7, j = rj % NT, R = rj / NT;
        const int row = (u * RG * RT + R) * 16 + (ln >> 2) + 8 * (c >> 1);
        const int col = 8 * j + 2 * (ln & 3) + (c & 1);
        if (col < m && row < n) out[(size_t)col * n + row] = f2bf(s);
      }
    }
    if (CS > 1) cluster_sync(); else __syncthreads();
  }
}
