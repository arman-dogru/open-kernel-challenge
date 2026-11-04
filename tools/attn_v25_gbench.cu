// Attention v25 gather microbenchmark (diagnostic only, not used by kernel_fn): the full-verify K/V gather alone
// (page-table rows, 64-key tiles of 128-byte K and V rows, ST-stage smem ring, mbarrier full / empty), consumers
// = 8 warps that only wait "full" and arrive "empty" (no math). Two producers:
//   gb_bulk: one producer warp, cp.async.bulk (TMA engine) of whole 128-byte rows, 128 per tile, linear rows
//            (no swizzle; complete_tx on "full", expect_tx by lane 0)
//   gb_cpa:  NPW producer warps, 16-byte cp.async (8 lanes per row, as attn_fvws), arrive.noinc on "full"
// Every inline asm touching shared memory carries a "memory" clobber.
#define BN 64
#define KROW 128
#define TILE (BN * KROW)
#ifndef ST
#define ST 3
#endif
#ifndef NPW
#define NPW 1
#endif
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  unsigned r;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void mbar_init(unsigned a, unsigned c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned a, unsigned par) {
  asm volatile("{\n .reg .pred p;\n GB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra GB_WAIT;\n}"
               :: "r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect(unsigned a, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk128(unsigned dst, const void* src, unsigned bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
               :: "r"(dst), "l"(src), "r"(bar) : "memory");
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

template <int BULK>
__device__ __forceinline__ void gb_body(const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
                                        const int* __restrict__ pt, int stride_pt, const int* __restrict__ seqlens,
                                        int chunk, int page_shift, unsigned* __restrict__ sink) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  const unsigned ks = smem_u32(smem), vs = ks + ST * TILE, bars = vs + ST * TILE;
  const int s = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int total = seqlens[b], lo = s * chunk, hi = min(lo + chunk, total);
  const int ntiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  const int* ptb = pt + (long long)b * stride_pt;
  const int pmask = (1 << page_shift) - 1;
  const int npw = BULK ? 1 : NPW;
  if (tid == 0) {
    for (int k = 0; k < ST; ++k) { mbar_init(bars + k * 8, BULK ? 1 : 32 * NPW); mbar_init(bars + (ST + k) * 8, 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp >= 8) {  // producer warps 8 .. 8 + npw - 1
    const int pwi = warp - 8;
    auto prow = [&](int tile, int k) -> int {
      const int t = lo + tile * BN + k;
      return (tile < ntiles && t < hi) ? (ptb[t >> page_shift] << page_shift) + (t & pmask) : -1;
    };
    int ra = prow(0, lane), rb = prow(0, lane + 32);
    int stg = 0, ph = 0;
    for (int i = 0; i < ntiles; ++i) {
      const int na = prow(i + 1, lane), nb = prow(i + 1, lane + 32);
      if (i >= ST) mbar_wait(bars + (ST + stg) * 8, ph ^ 1);
      const unsigned kd = ks + stg * TILE, vd = vs + stg * TILE, fb = bars + stg * 8;
      if (BULK) {
        if (lane == 0) mbar_expect(fb, 2 * TILE);
        __syncwarp();
        const long long r0 = (long long)(ra >= 0 ? ra : 0) * KROW, r1 = (long long)(rb >= 0 ? rb : 0) * KROW;
        bulk128(kd + lane * KROW, kc + r0, fb);
        bulk128(vd + lane * KROW, vc + r0, fb);
        bulk128(kd + (lane + 32) * KROW, kc + r1, fb);
        bulk128(vd + (lane + 32) * KROW, vc + r1, fb);
      } else {
        const int pk = lane >> 3, pc = lane & 7;
#pragma unroll
        for (int jj = 0; jj < 16 / NPW; ++jj) {
          const int j = pwi * (16 / NPW) + jj;
          const int key = pk + 4 * j;
          const int row = __shfl_sync(0xffffffffu, j < 8 ? ra : rb, key & 31);
          const unsigned off = key * KROW + ((pc ^ (key & 7)) << 4);
          const int n = row >= 0 ? 16 : 0;
          const long long r = (long long)(row >= 0 ? row : 0) * KROW + pc * 16;
          cp16(kd + off, kc + r, n);
          cp16(vd + off, vc + r, n);
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(fb) : "memory");
      }
      ra = na; rb = nb;
      if (++stg == ST) { stg = 0; ph ^= 1; }
    }
    if (!BULK) asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }
  unsigned acc = 0;
  int st = 0, ph = 0;
  for (int i = 0; i < ntiles; ++i) {
    mbar_wait(bars + st * 8, ph);
    unsigned w;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w) : "r"(ks + st * TILE + tid * 4) : "memory");
    acc += w;
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + (ST + st) * 8);
    if (++st == ST) { st = 0; ph ^= 1; }
  }
  if (acc == 0x12345678u) sink[0] = acc;
}
extern "C" __global__ void gb_bulk(const unsigned char* kc, const unsigned char* vc, const int* pt, int stride_pt,
                                   const int* seqlens, int chunk, int page_shift, unsigned* sink) {
  gb_body<1>(kc, vc, pt, stride_pt, seqlens, chunk, page_shift, sink);
}
extern "C" __global__ void gb_cpa(const unsigned char* kc, const unsigned char* vc, const int* pt, int stride_pt,
                                  const int* seqlens, int chunk, int page_shift, unsigned* sink) {
  gb_body<0>(kc, vc, pt, stride_pt, seqlens, chunk, page_shift, sink);
}
