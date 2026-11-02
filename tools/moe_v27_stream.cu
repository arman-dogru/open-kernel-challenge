// MoE v27: weight-streaming grouped NVFP4 x BF16 expert GEMM (routed experts, <= 32 tokens), sm_90a, mma.sync
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
#pragma unroll
    for (int j = 0; j < (NOCOMPUTE ? 0 : TPS / NW); ++j) {  // TPS % NW == 0: fixed trip count, fully unrolled
      const int i = warp + j * NW;
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
