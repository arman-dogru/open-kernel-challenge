// bf16 v31: weight-streaming skinny GEMM out[m, n] = x[m, K] . W[n, K]^T for m <= 32 (BF16 in, FP32 accumulate,
// BF16 out; FP32 out for the router's hi/lo pair), sm_90a, mma.sync m16n8k16 swap-AB (W rows = MMA M, tokens = N).
// Why not v30: v30's register loads (LDG.128 interleaved with their MMAs by ptxas, ~2 per warp in flight) and an
// unbalanced grid (qkv_gate: 144 rowtiles on 132 SMs) kept it at ~2.6-2.9 TB/s. v31 instead:
// - one producer warp per CTA streams W with TMA bulk copies (cp.async.bulk, L2 evict-first) into an ST-deep ring of
//   SPS-KB stages (64-128 KB in flight per SM), mbarrier full/empty handshakes, NW consumer warps read A fragments
//   from shared memory (contiguous 16 B per lane, conflict-free) and run the MMAs.
// - W repacked (prepare) into A-fragment order [rowtile (16 rows)][pair (32 k)][step s][lane][8 bf16] (the v30
//   layout); k permuted inside each pair (lane t, step s holds k = 8t + 4s + 0..3) so x's B fragments for both steps
//   are one 16 B shared-memory load.
// - balanced 1-CTA-per-SM grid of exactly (n/16/G) * CS CTAs (<= 132): a cluster of CS CTAs owns G rowtiles; rank r
//   owns k-slice r (K/CS) of all G rowtiles, so its x slice is bulk-copied into shared memory once and reused G times.
// - split-K reduction without atomics, fixed order: consumer warps -> shared memory per rowtile, then the CS ranks'
//   partials through distributed shared memory (each output rowtile summed by one owner rank, ranks in order).
//   HL = 2 sums rowtile pairs (router: rowtile 2c = hi rows, 2c+1 = lo rows of output rowtile c). Deterministic.
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
__device__ __forceinline__ u32 cluster_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_dsmem(const float* p, u32 rank) {
  u32 ra;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(sa(p)), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}
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

// Shared memory (dynamic, host computes the same sizes in bf16_v31_cfg.py smem()):
//   ring [ST][SPS KB] | xs [8 NT rows][XROW = Pc*64 + 64 B] | red [2][NW][E] f32 | seg [G][E] f32 (CS > 1 or HL > 1)
//   | mbarriers full[ST], empty[ST], x
// Launch: grid (n_rows/16/G) * CS CTAs, (NW + 1) * 32 threads. Pc = K/32/CS pairs per rank per rowtile.
template <int NT, int CS, int NW, int ST, int SPS, int HL, bool F32>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__((NW + 1) * 32, 1)
bf16_v31(const unsigned char* __restrict__ W, const u16* __restrict__ x, int ldx_, void* __restrict__ out, int m,
         int n_out, int K, int G) {
  constexpr int E = NT * 128;  // FP32 partials per rowtile: 16 rows x 8 NT tokens
  constexpr bool SEG = CS > 1 || HL > 1;
  constexpr u32 SB = SPS * 1024;
  extern __shared__ __align__(128) unsigned char sm[];
  const int PPR = K >> 5, Pc = PPR / CS, nst = Pc / SPS;
  const int XROW = Pc * 64 + 64;
  unsigned char* ring = sm;
  unsigned char* xs = sm + ST * SB;
  float* red = reinterpret_cast<float*>(xs + 8 * NT * XROW);
  float* seg = red + 2 * NW * E;
  u64* bars = reinterpret_cast<u64*>(seg + (SEG ? G * E : 0));
  u64* full = bars;
  u64* empty = bars + ST;
  u64* xbar = bars + 2 * ST;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const u32 rank = CS > 1 ? cluster_rank() : 0u;
  const int cl = blockIdx.x / CS;
  const int rt0 = cl * G;              // first rowtile of this cluster
  const int kp0 = (int)rank * Pc;      // first pair of this rank's k-slice
  if (threadIdx.x == 0) {
    for (int i = 0; i < ST; ++i) {
      mbar_init(full + i, 1);
      mbar_init(empty + i, NW);
    }
    mbar_init(xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == NW) {  // producer
    if (lane == 0) {
      u64 pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      mbar_expect_tx(xbar, (u32)m * (u32)(Pc * 64));
      for (int r = 0; r < m; ++r)
        bulk_g2s(xs + r * XROW, x + (size_t)r * ldx_ + kp0 * 32, (u32)(Pc * 64), xbar);
      int it = 0;
      for (int g = 0; g < G; ++g) {
        const unsigned char* src = W + ((size_t)(rt0 + g) * PPR + kp0) * 1024;
        for (int c = 0; c < nst; ++c, ++it) {
          const int s = it % ST;
          if (it >= ST) mbar_wait(empty + s, (u32)((it / ST) - 1) & 1u);
          mbar_expect_tx(full + s, SB);
          bulk_g2s_ef(ring + s * SB, src + (size_t)c * SB, SB, full + s, pol);
        }
      }
    }
  } else {  // consumers
    const int t = lane & 3, gq = lane >> 2;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
    mbar_wait(xbar, 0);
    const unsigned char* xl = xs + gq * XROW + 16 * t;
    int it = 0;
    for (int g = 0; g < G; ++g) {
      for (int c = 0; c < nst; ++c, ++it) {
        const int s = it % ST;
        mbar_wait(full + s, (u32)(it / ST) & 1u);
        const unsigned char* A = ring + s * SB + lane * 16;
#pragma unroll 4
        for (int i = warp; i < SPS; i += NW) {
          const uint4 a0 = lds128(A + i * 1024), a1 = lds128(A + i * 1024 + 512);
          const unsigned char* xb = xl + (c * SPS + i) * 64;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint4 b = lds128(xb + j * 8 * XROW);
            mma(acc[j], a0.x, a0.y, a0.z, a0.w, b.x, b.y);
            mma(acc[j], a1.x, a1.y, a1.z, a1.w, b.z, b.w);
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + s);
      }
      // rowtile g done: reduce the NW warps' partials in fixed order (double-buffered: one barrier per rowtile)
      float* rb = red + (g & 1) * NW * E;
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          rb[warp * E + (j * 4 + c) * 32 + lane] = acc[j][c];
          acc[j][c] = 0.f;
        }
      bar_consumers(NW * 32);
      for (int e = threadIdx.x; e < E; e += NW * 32) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) v += rb[w * E + e];
        if (SEG) {
          seg[g * E + e] = v;
        } else {
          const int ln = e & 31, c = (e >> 5) & 3, j = e >> 7;
          const int row = (rt0 + g) * 16 + (ln >> 2) + 8 * (c >> 1);
          const int col = 8 * j + 2 * (ln & 3) + (c & 1);
          if (col < m) {
            if (F32) reinterpret_cast<float*>(out)[(size_t)col * n_out + row] = v;
            else reinterpret_cast<u16*>(out)[(size_t)col * n_out + row] = f2bf(v);
          }
        }
      }
    }
  }
  __syncwarp();
  if (SEG) {
    if (CS > 1) cluster_sync(); else __syncthreads();
    const int GO = G / HL;  // output rowtiles of this cluster
    for (int o = (int)rank; o < GO; o += CS) {
      for (int e = threadIdx.x; e < E; e += (NW + 1) * 32) {
        float v = 0.f;
#pragma unroll
        for (int r = 0; r < CS; ++r)
#pragma unroll
          for (int h = 0; h < HL; ++h) v += CS > 1 ? ld_dsmem(seg + (o * HL + h) * E + e, (u32)r) : seg[(o * HL + h) * E + e];
        const int ln = e & 31, c = (e >> 5) & 3, j = e >> 7;
        const int row = (cl * GO + o) * 16 + (ln >> 2) + 8 * (c >> 1);
        const int col = 8 * j + 2 * (ln & 3) + (c & 1);
        if (col < m) {
          if (F32) reinterpret_cast<float*>(out)[(size_t)col * n_out + row] = v;
          else reinterpret_cast<u16*>(out)[(size_t)col * n_out + row] = f2bf(v);
        }
      }
    }
    if (CS > 1) cluster_sync();
  }
}
