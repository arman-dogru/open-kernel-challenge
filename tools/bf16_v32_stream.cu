// bf16 v32: weight-streaming skinny GEMM out[m, n] = x[m, K] . W[n, K]^T for m <= 32 (BF16 in, FP32 accumulate,
// BF16 out; FP32 out for the router's hi/lo pair), sm_90a, mma.sync m16n8k16 swap-AB (W rows = MMA M, tokens = N).
// v31 -> v32 (v31d diag cab5ba2512714869): clusters of 8 / 4 CTAs at ~150 KB smem are only 15 / 30 co-resident on
// the H200 (16 / 32 needed), so v31's qkv_gate / router / CS>=4 plans ran in 2 waves. v32 drops clusters:
// - grid = (rows/16/G) * KS CTAs (<= 132, one per SM: >= 116 KB smem): group q of KS CTAs owns rowtiles
//   [qG, qG+G); CTA ks of the group owns k-slice ks (K/KS) of all of them, so its x slice is bulk-copied into shared
//   memory once (per stage chunk, so the first rowtile starts after one chunk) and reused G times.
// - one producer warp streams W with TMA bulk copies (cp.async.bulk, L2 evict-first) into an ST-deep ring of stages
//   of RT rowtiles x SPS pairs (1 KB each); NW consumer warps read A fragments from shared memory (16 B per lane,
//   conflict-free) and B fragments from the x slice (rows padded by 64 B: conflict-free), RT rowtiles per B load.
// - W repacked (prepare) into A-fragment order [rowtile (16 rows)][pair (32 k)][step s][lane][8 bf16]; k permuted
//   inside each pair (lane t, step s holds k = 8t + 4s + 0..3) so both steps' B fragments are one 16 B load.
// - split-K reduction without atomics, fixed order: warps -> shared memory per rowtile; KS > 1: every CTA writes its
//   G partials to a global workspace, then publishes a per-CTA launch counter (st.release.gpu; the CTA is its only
//   writer: new = old + 1, all counters of a grid move in lockstep), and output rowtile o's owner (ks == o % KS)
//   waits for its KS peers' counters (ld.acquire.gpu) and sums their partials in ks order (and the HL hi/lo pair).
//   All CTAs are co-resident (grid <= SMs, one per SM), so the wait cannot deadlock; a bounded spin traps anyway.
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;

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
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, u32 bytes, u64* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s_ef(void* dst, const void* src, u32 bytes, u64* bar, u64 pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(sa(p))
               : "memory");
  return v;
}
__device__ __forceinline__ void bar_consumers(int nthreads) {
  asm volatile("bar.sync 1, %0;" :: "r"(nthreads) : "memory");
}


__device__ __forceinline__ void spin_ge(const u32* p, u32 want) {
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
__device__ __forceinline__ void put(void* out, size_t i, float v, bool f32) {
  if (f32) reinterpret_cast<float*>(out)[i] = v;
  else reinterpret_cast<u16*>(out)[i] = f2bf(v);
}

// Shared memory (dynamic; host: bf16_v32_cfg.py smem()):
//   ring [ST][RT][SPS KB] | xs [8 NT rows][XROW = Pc*64 + 64 B] | red [2][NW][RT*E] f32 | mbarriers full[ST],
//   empty[ST], x[16]. Launch: grid (rows/16/G) * KS CTAs, (NW + 1) * 32 threads. Pc = K/32/KS pairs per CTA.
// ws: [grid][G][E] f32 partials (KS > 1 or HL > 1), cnt: [grid] u32 launch counters (zeroed once in prepare()).
template <int NT, int RT, int NW, int ST, int SPS, int HL, bool F32>
__global__ void __launch_bounds__((NW + 1) * 32, 1)
bf16_v32(const unsigned char* __restrict__ W, const u16* __restrict__ x, int ldx_, void* __restrict__ out, int m,
         int n_out, int K, int G, int KS, float* __restrict__ ws, u32* __restrict__ cnt) {
  constexpr int E = NT * 128;  // FP32 partials per rowtile: 16 rows x 8 NT tokens
  constexpr u32 SB = SPS * 1024;
  extern __shared__ __align__(128) unsigned char sm[];
  const int PPR = K >> 5, Pc = PPR / KS, nst = Pc / SPS;
  const int XROW = Pc * 64 + 64;
  const bool SEG = KS > 1 || HL > 1;
  unsigned char* ring = sm;
  unsigned char* xs = sm + ST * RT * SB;
  float* red = reinterpret_cast<float*>(xs + 8 * NT * XROW);
  u64* bars = reinterpret_cast<u64*>(red + 2 * NW * RT * E);
  u64* full = bars;
  u64* empty = bars + ST;
  u64* xbar = bars + 2 * ST;  // [16], one per stage chunk of the x slice
  __shared__ u32 s_old;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int grp = blockIdx.x / KS, ks = blockIdx.x % KS;
  const int rt0 = grp * G;   // first rowtile of this group
  const int kp0 = ks * Pc;   // first pair of this CTA's k-slice
  if (threadIdx.x == 0) {
    for (int i = 0; i < ST; ++i) {
      mbar_init(full + i, 1);
      mbar_init(empty + i, NW);
    }
    for (int i = 0; i < 16; ++i) mbar_init(xbar + i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (SEG) s_old = *reinterpret_cast<volatile u32*>(cnt + blockIdx.x);
  }
  __syncthreads();
  if (warp == NW) {  // producer
    if (lane == 0) {
      u64 pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      int it = 0;
      for (int g = 0; g < G; g += RT) {
        for (int c = 0; c < nst; ++c, ++it) {
          if (g == 0) {  // x chunk c (needed from stage c of the first rowtile group on)
            mbar_expect_tx(xbar + c, (u32)m * (u32)(SPS * 64));
            for (int r = 0; r < m; ++r)
              bulk_g2s(xs + r * XROW + c * SPS * 64, x + (size_t)r * ldx_ + (kp0 + c * SPS) * 32, (u32)(SPS * 64),
                       xbar + c);
          }
          const int s = it % ST;
          if (it >= ST) mbar_wait(empty + s, (u32)((it / ST) - 1) & 1u);
          mbar_expect_tx(full + s, RT * SB);
#pragma unroll
          for (int r = 0; r < RT; ++r)
            bulk_g2s_ef(ring + (s * RT + r) * SB, W + ((size_t)(rt0 + g + r) * PPR + kp0 + c * SPS) * 1024, SB,
                        full + s, pol);
        }
      }
    }
  } else {  // consumers
    const int t = lane & 3, gq = lane >> 2;
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][j][c] = 0.f;
    const unsigned char* xl = xs + gq * XROW + 16 * t;
    int it = 0;
    for (int g = 0; g < G; g += RT) {
      for (int c = 0; c < nst; ++c, ++it) {
        const int s = it % ST;
        if (g == 0) mbar_wait(xbar + c, 0);
        mbar_wait(full + s, (u32)(it / ST) & 1u);
        const unsigned char* A = ring + s * RT * SB + lane * 16;
#pragma unroll 2
        for (int i = warp; i < SPS; i += NW) {
          uint4 a0[RT], a1[RT];
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            a0[r] = lds128(A + r * SB + i * 1024);
            a1[r] = lds128(A + r * SB + i * 1024 + 512);
          }
          const unsigned char* xb = xl + (c * SPS + i) * 64;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint4 b = lds128(xb + j * 8 * XROW);
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              mma(acc[r][j], a0[r].x, a0[r].y, a0[r].z, a0[r].w, b.x, b.y);
              mma(acc[r][j], a1[r].x, a1[r].y, a1[r].z, a1[r].w, b.z, b.w);
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + s);
      }
      // rowtiles g..g+RT-1 done: reduce the NW warps' partials in fixed order (double-buffered: one barrier each)
      float* rb = red + ((g / RT) & 1) * NW * RT * E;
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            rb[warp * RT * E + r * E + (j * 4 + c) * 32 + lane] = acc[r][j][c];
            acc[r][j][c] = 0.f;
          }
      bar_consumers(NW * 32);
      for (int e = threadIdx.x; e < RT * E; e += NW * 32) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) v += rb[w * RT * E + e];
        const int r = e / E, ee = e % E;
        if (SEG) {
          ws[((size_t)blockIdx.x * G + g + r) * E + ee] = v;
        } else {
          const int ln = ee & 31, c = (ee >> 5) & 3, j = ee >> 7;
          const int row = (rt0 + g + r) * 16 + (ln >> 2) + 8 * (c >> 1);
          const int col = 8 * j + 2 * (ln & 3) + (c & 1);
          if (col < m) put(out, (size_t)col * n_out + row, v, F32);
        }
      }
    }
  }
  if (SEG) {
    __threadfence();
    __syncthreads();
    const u32 want = s_old + 1u;
    if (threadIdx.x == 0) st_release(cnt + blockIdx.x, want);
    const int GO = G / HL;  // output rowtiles of this group
    if (ks < GO) {
      if (threadIdx.x == 0)
        for (int r = 0; r < KS; ++r) spin_ge(cnt + grp * KS + r, want);
      __syncthreads();
      for (int o = ks; o < GO; o += KS) {
        for (int e = threadIdx.x; e < E; e += (NW + 1) * 32) {
          float v = 0.f;
          for (int r = 0; r < KS; ++r)
#pragma unroll
            for (int h = 0; h < HL; ++h) v += __ldcg(ws + ((size_t)(grp * KS + r) * G + o * HL + h) * E + e);
          const int ln = e & 31, c = (e >> 5) & 3, j = e >> 7;
          const int row = (grp * GO + o) * 16 + (ln >> 2) + 8 * (c >> 1);
          const int col = 8 * j + 2 * (ln & 3) + (c & 1);
          if (col < m) put(out, (size_t)col * n_out + row, v, F32);
        }
      }
    }
  }
}
