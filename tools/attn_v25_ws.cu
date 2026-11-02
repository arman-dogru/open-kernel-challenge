// Full-attention verify (q_len 8) split-KV partials, attention v25 warp-specialized variant (attn_fvws).
// Same math, layouts and outputs as attn_v25_fv.cu (v24's FV kernel), but the K/V gather is decoupled from the
// compute: 288 threads = 2 consumer warpgroups (one per KV head, as v24) + 1 producer warp.
//  - producer warp: per 64-key tile, resolves the page-table rows (one tile ahead, shuffled across lanes), waits for
//    the stage's "empty" mbarrier, issues 32 cp.async per lane (8 lanes per 128-byte K / V row: 4 whole rows per
//    instruction) into the 128-byte-swizzled stage, then cp.async.mbarrier.arrive.noinc on the stage's "full"
//    mbarrier (count 32), so a stage turns full when every lane's copies have landed.
//  - consumers: wait "full", fence.proxy.async (cp.async writes -> wgmma async proxy), named barrier per
//    warpgroup (bar 1 + h, 128 threads: every warp of the warpgroup is past tile i - 1's waits, so V^T buffer i & 1
//    is free), Q.K issue, V transpose of the warpgroup's own head only, fence + named barrier, wait S(i) / P.V(i - 1),
//    then each warp's lane 0 arrives on the stage's "empty" mbarrier (count 8), softmax, P.V issue.
//  - v24 diagnostic (v25d4): loads alone ran at ~2.85 TB/s but the full kernel at ~2 TB/s because every tile's
//    cp.async issue waited behind the CTA-wide barriers and the softmax; here no consumer barrier gates the loads.
// The last tile's P.V is waited inside the loop (no wgmma serialization, ptxas C7515).
// Every inline asm touching shared memory carries a "memory" clobber.
#define D 64
#define HKV 2
#define G 8
#define LQ 8
#define BN 64
#ifndef ST
#define ST 3  // cp.async stages: ST - 1 tiles (16 KB each) in flight per CTA
#endif
#define QH 8192  // Q bytes per KV head in shared memory: 64 rows x 128 B (64 used), 128-byte swizzle
#define KROW 128
#define VTH 9216  // V^T bytes per head: 64 dims + 8 rows of ones (row sums of P) x 128-byte rows, 128-byte swizzle
#define TILE (BN * KROW)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  unsigned r;
  asm volatile("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ unsigned long long desc(unsigned addr, unsigned sbo, unsigned mode) {
  return (unsigned long long)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(sbo >> 4) << 32) |
         ((unsigned long long)mode << 62);
}
#define WG_PV2(acc, a, b, d0, d1)                                                                           \
  asm volatile("wgmma.fence.sync.aligned;\n"                                                                   \
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %44, 1, 1, 1;\n" \
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%40,%41,%42,%43}, %45, 1, 1, 1;\n" \
               "wgmma.commit_group.sync.aligned;"                                                                 \
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3])                                                                         \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),         \
                 "l"(d0), "l"(d1) : "memory")
#define WG_QK_SS(acc, da, db, sc)                                                                          \
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"                                                        \
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13," \
               "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"       \
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]),              \
                 "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]),              \
                 "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]),              \
                 "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]),              \
                 "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]),              \
                 "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]),              \
                 "+f"(acc[7][2]), "+f"(acc[7][3])                                                                 \
               : "l"(da), "l"(db), "r"(sc))
#define WG_WAIT_SO(sacc, oacc)                                                                              \
  asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(sacc[0][0]), "+f"(sacc[0][1]), "+f"(sacc[0][2]), "+f"(sacc[0][3]), "+f"(sacc[1][0]), "+f"(sacc[1][1]), "+f"(sacc[1][2]), "+f"(sacc[1][3]), "+f"(sacc[2][0]), "+f"(sacc[2][1]), "+f"(sacc[2][2]), "+f"(sacc[2][3]), "+f"(sacc[3][0]), "+f"(sacc[3][1]), "+f"(sacc[3][2]), "+f"(sacc[3][3]), "+f"(sacc[4][0]), "+f"(sacc[4][1]), "+f"(sacc[4][2]), "+f"(sacc[4][3]), "+f"(sacc[5][0]), "+f"(sacc[5][1]), "+f"(sacc[5][2]), "+f"(sacc[5][3]), "+f"(sacc[6][0]), "+f"(sacc[6][1]), "+f"(sacc[6][2]), "+f"(sacc[6][3]), "+f"(sacc[7][0]), "+f"(sacc[7][1]), "+f"(sacc[7][2]), "+f"(sacc[7][3]), "+f"(oacc[0][0]), "+f"(oacc[0][1]), "+f"(oacc[0][2]), "+f"(oacc[0][3]), "+f"(oacc[1][0]), "+f"(oacc[1][1]), "+f"(oacc[1][2]), "+f"(oacc[1][3]), "+f"(oacc[2][0]), "+f"(oacc[2][1]), "+f"(oacc[2][2]), "+f"(oacc[2][3]), "+f"(oacc[3][0]), "+f"(oacc[3][1]), "+f"(oacc[3][2]), "+f"(oacc[3][3]), "+f"(oacc[4][0]), "+f"(oacc[4][1]), "+f"(oacc[4][2]), "+f"(oacc[4][3]), "+f"(oacc[5][0]), "+f"(oacc[5][1]), "+f"(oacc[5][2]), "+f"(oacc[5][3]), "+f"(oacc[6][0]), "+f"(oacc[6][1]), "+f"(oacc[6][2]), "+f"(oacc[6][3]), "+f"(oacc[7][0]), "+f"(oacc[7][1]), "+f"(oacc[7][2]), "+f"(oacc[7][3]), "+f"(oacc[8][0]), "+f"(oacc[8][1]), "+f"(oacc[8][2]), "+f"(oacc[8][3]) :: "memory")
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(unsigned addr, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned addr, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n WS_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WS_WAIT;\n}"
               :: "r"(addr), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void cp_arrive_noinc(unsigned addr) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void wg_bar(int h) {  // named barrier 1 + h over warpgroup h (immediate ids: ptxas reserves 2)
  if (h) asm volatile("bar.sync 2, 128;" ::: "memory");
  else asm volatile("bar.sync 1, 128;" ::: "memory");
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned pack_e4(float x0, float x1, float x2, float x3) {  // bytes x0 (low) .. x3
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (unsigned)lo | ((unsigned)hi << 16);
}

extern "C" __global__ void __maxnreg__(112)  // 2 CTAs/SM: 9 warps x 2 x 32 x 112 <= 64K registers
attn_fvws(const unsigned char* __restrict__ q, int stride_qt, const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc, const int* __restrict__ pt, int stride_pt,
        const int* __restrict__ seqlens, const float* __restrict__ ksp, const float* __restrict__ vsp,
        float* __restrict__ o_part, float* __restrict__ lse_out, float qk_scale, int chunk, int page_shift,
        int nsplit, int hq) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);  // 1024-aligned (SW128 atoms)
  unsigned char* Ks = smem;
  unsigned char* Vs = smem + ST * TILE;
  unsigned char* Vt = smem + 2 * ST * TILE;
  unsigned char* Qs = Vt + 2 * 2 * VTH;  // V^T double-buffered per head (P.V of tile i runs into tile i + 1)
  const unsigned bars = smem_u32(Qs + 2 * QH);  // full[ST], empty[ST] mbarriers (8 B each)
#define FULL(k) (bars + (k) * 8)
#define EMPTY(k) (bars + (ST + (k)) * 8)
  const float NEG = __int_as_float(0xff800000);
  const int s = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = warp >> 2, wi = warp & 3;
  const int g = lane >> 2, t4 = lane & 3;
  const int total = seqlens[b];
  const int seq = total - LQ;
  const int lo = s * chunk;
  const int hi = min(lo + chunk, total);
  const int ntiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  const int j0 = wi * 2, j1 = wi * 2 + 1;           // rows wi*16 + g and + 8: tokens j0 / j1, query head h*8 + g
  const int pos0 = seq + j0, pos1 = seq + j1;
  const int head = h * G + g;

  if (tid < 256) {  // Q -> shared memory once (wgmma A operand by descriptor). Row m of KV head hh's 64 = token 2 (m >> 4) +
     // ((m >> 3) & 1), query head hh * 8 + (m & 7): warp wi's accumulator rows wi * 16 + g / + 8 are tokens j0 / j1.
    const int row = tid >> 1, hh = row >> 6, m = row & 63;
    const int tok = 2 * (m >> 4) + ((m >> 3) & 1), qh = hh * G + (m & 7);
    const unsigned char* src = q + (long long)(b * LQ + tok) * stride_qt + qh * D;
#pragma unroll
    for (int c = (tid & 1) * 2; c < (tid & 1) * 2 + 2; ++c)
      *(uint4*)(Qs + hh * QH + m * 128 + ((c ^ (m & 7)) << 4)) = *(const uint4*)(src + c * 16);
  }
  const float scale = qk_scale * ksp[0] * 1.4426950408889634f;
  const float vsc = vsp[0];

  // Per-thread shared-memory offsets, computed once and pinned in registers (v23: v22 recomputed all of this
  // address math from threadIdx every tile).
  // V -> V^T transpose: per rep, loads at tr_l[rep] (+1024) and tr_l1[rep] (+1024), stores at tr_s[rep][0..3]
  unsigned tr_l[2], tr_l1[2], tr_s[2][4];
#pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    // v24's task map with warp -> pw: warpgroup h covers dims of its own head only (dg in 16h .. 16h + 15)
    const int pw = 2 * h + (wi & 1) + 4 * (wi >> 1);
    const int combo = pw + 8 * rep;
    const int dg = (lane & 7) + 8 * (combo & 3), sg = (lane >> 3) + 4 * (combo >> 2);
    const int blk = sg >> 3, sl = sg & 7, tt = sl & 3, A = sl >> 2;
    const int k0 = blk * 32 + 16 * A + 2 * tt;  // keys k0, k0 + 1, k0 + 8, k0 + 9 (k0 & 7 == 2 tt)
    tr_l[rep] = k0 * KROW + ((((dg >> 2) ^ (k0 & 7)) * 16) + (dg & 3) * 4);
    tr_l1[rep] = (k0 + 1) * KROW + ((((dg >> 2) ^ ((k0 + 1) & 7)) * 16) + (dg & 3) * 4);
    const int slot = blk * 32 + sl * 4;
#pragma unroll
    for (int r = 0; r < 4; ++r) {  // row = head * 64 + d; 128-byte rows, 16-byte chunk ^= row & 7
      const int row = dg * 4 + r;
      tr_s[rep][r] = (row >> 6) * VTH + (row & 63) * 128 + (((slot >> 4) ^ (row & 7)) << 4) + (slot & 15);
    }
  }
  asm volatile("" : "+r"(tr_l[0]), "+r"(tr_l[1]), "+r"(tr_l1[0]), "+r"(tr_l1[1]));
  asm volatile("" : "+r"(tr_s[0][0]), "+r"(tr_s[0][1]), "+r"(tr_s[0][2]), "+r"(tr_s[0][3]), "+r"(tr_s[1][0]),
               "+r"(tr_s[1][1]), "+r"(tr_s[1][2]), "+r"(tr_s[1][3]));
  const int* ptb = pt + (long long)b * stride_pt;
  const int pmask = (1 << page_shift) - 1;
  float oacc[9][4];  // [0..7]: O (64 dims); [8]: sum of P over keys (V^T rows 64..71 are ones)
#pragma unroll
  for (int n = 0; n < 9; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
  float m0 = NEG, m1 = NEG;
  for (int k = tid; k < 2 * 2 * 8 * 8; k += 288) {  // ones rows: 2 buffers x 2 heads x 8 rows x 8 chunks (E4M3 1.0)
    const int buf_head = k >> 6, rr = (k >> 3) & 7, c = k & 7;
    *(uint4*)(Vt + buf_head * VTH + (64 + rr) * 128 + c * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }

  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < ST; ++k) { mbar_init(FULL(k), 32); mbar_init(EMPTY(k), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // Q / ones rows (st.shared) -> wgmma
  __syncthreads();

  if (warp == 8) {  // producer warp
    const int pk = lane >> 3, pc = lane & 7;  // instruction j: key pk + 4 j, 16-byte chunk pc of its K and V rows
    auto prow = [&](int tile, int k) -> int {
      const int t = lo + tile * BN + k;
      return (tile < ntiles && t < hi) ? (ptb[t >> page_shift] << page_shift) + (t & pmask) : -1;
    };
    int ra = prow(0, lane), rb = prow(0, lane + 32);  // rows of keys lane / lane + 32 of the tile
    const unsigned ks_base = smem_u32(Ks), vs_base = smem_u32(Vs);
    int stg = 0, ph = 0;
    for (int i = 0; i < ntiles; ++i) {
      const int na = prow(i + 1, lane), nb = prow(i + 1, lane + 32);  // next tile's rows, in flight meanwhile
      if (i >= ST) mbar_wait(EMPTY(stg), ph ^ 1);  // tile i - ST (same stage) consumed by all 8 consumer warps
      const unsigned kd = ks_base + stg * TILE, vd = vs_base + stg * TILE;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int key = pk + 4 * j;
        const int row = __shfl_sync(0xffffffffu, j < 8 ? ra : rb, key & 31);
        const unsigned off = key * KROW + ((pc ^ (key & 7)) << 4);
        const int n = row >= 0 ? 16 : 0;
        const long long r = (long long)(row >= 0 ? row : 0) * KROW + pc * 16;
        cp16(kd + off, kc + r, n);
        cp16(vd + off, vc + r, n);
      }
      cp_arrive_noinc(FULL(stg));
      ra = na; rb = nb;
      if (++stg == ST) { stg = 0; ph ^= 1; }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }

  const unsigned qs = smem_u32(Qs + h * QH);
  const unsigned ks_base = smem_u32(Ks), vs_base = smem_u32(Vs), vt_base = smem_u32(Vt);
  int st = 0, ph = 0;  // stage of tile i and its "full" phase parity
  for (int i = 0; i < ntiles; ++i) {
    mbar_wait(FULL(st), ph);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // producer's cp.async data -> wgmma (async proxy)
    wg_bar(h);  // every warp of this warpgroup is past tile i - 1's waits: V^T buffer i & 1 (P.V(i - 2)) free

    // S(i) = Q.K^T issued first, so the tensor cores run it while this warpgroup transposes V(i)
    const unsigned kt = ks_base + st * TILE + h * 64;
    float sc[8][4] = {};
    wg_fence();
    WG_QK_SS(sc, desc(qs, 1024, 1), desc(kt, 1024, 1), 0);
    WG_QK_SS(sc, desc(qs + 32, 1024, 1), desc(kt + 32, 1024, 1), 1);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    {  // V tile i -> V^T[i & 1]: task = 4 dims x 4 key slots; slot 4t + 2a' + e of 32-key block <- key 8(2A + a') + 2t + e
      const unsigned vsrc = vs_base + st * TILE, vdst = vt_base + (i & 1) * 2 * VTH;
#pragma unroll
      for (int rep = 0; rep < 2; ++rep) {
        unsigned w0, w1, w2, w3;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w0) : "r"(vsrc + tr_l[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w1) : "r"(vsrc + tr_l1[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1+1024];" : "=r"(w2) : "r"(vsrc + tr_l[rep]) : "memory");
        asm volatile("ld.shared.b32 %0, [%1+1024];" : "=r"(w3) : "r"(vsrc + tr_l1[rep]) : "memory");
        const unsigned x = __byte_perm(w0, w1, 0x5140), y = __byte_perm(w2, w3, 0x5140);
        const unsigned x2 = __byte_perm(w0, w1, 0x7362), y2 = __byte_perm(w2, w3, 0x7362);
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][0]), "r"(__byte_perm(x, y, 0x5410)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][1]), "r"(__byte_perm(x, y, 0x7632)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][2]), "r"(__byte_perm(x2, y2, 0x5410)) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(vdst + tr_s[rep][3]), "r"(__byte_perm(x2, y2, 0x7632)) : "memory");
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // V^T (st.shared) -> P.V wgmma (async proxy)
    wg_bar(h);
    WG_WAIT_SO(sc, oacc);  // S(i) and the previous tile's P.V (overlapped with this tile's barrier/transpose)
    if (lane == 0) mbar_arrive(EMPTY(st));  // this warp is done with stage st (its K via S(i), its V via the transpose)
    if (++st == ST) { st = 0; ph ^= 1; }
    const int base = lo + i * BN;
    if (!(base + BN <= hi && base + BN - 1 <= pos0)) {  // edge tile: mask (warp-uniform branch)
      const int tb = base + t4 * 2;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int t = tb + n * 8 + e;
          if (!(t < hi && t <= pos0)) sc[n][e] = NEG;
          if (!(t < hi && t <= pos1)) sc[n][2 + e] = NEG;
        }
      }
    }
    float mx0 = NEG, mx1 = NEG;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      mx0 = fmaxf(mx0, fmaxf(sc[n][0], sc[n][1])); mx1 = fmaxf(mx1, fmaxf(sc[n][2], sc[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m0, mx0 * scale), mn1 = fmaxf(m1, mx1 * scale);
    const float ms0 = mn0 == NEG ? 0.f : mn0, ms1 = mn1 == NEG ? 0.f : mn1;
    const float a0 = ex2(m0 - ms0), a1 = ex2(m1 - ms1);
    const float nb0 = 8.f - ms0, nb1 = 8.f - ms1;  // p = 2^(s - m + 8) in (0, 256]
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sc[n][0] = ex2(fmaf(sc[n][0], scale, nb0)); sc[n][1] = ex2(fmaf(sc[n][1], scale, nb0));
      sc[n][2] = ex2(fmaf(sc[n][2], scale, nb1)); sc[n][3] = ex2(fmaf(sc[n][3], scale, nb1));
    }
    if (!__all_sync(0xffffffff, mn0 == m0 && mn1 == m1)) {  // rescale O (and the row sums) only if a max moved
#pragma unroll
      for (int n = 0; n < 9; ++n) { oacc[n][0] *= a0; oacc[n][1] *= a0; oacc[n][2] *= a1; oacc[n][3] *= a1; }
    }
    m0 = mn0; m1 = mn1;
    unsigned pa[2][4];
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const int n0 = kb * 4;
      pa[kb][0] = pack_e4(sc[n0][0], sc[n0][1], sc[n0 + 1][0], sc[n0 + 1][1]);
      pa[kb][1] = pack_e4(sc[n0][2], sc[n0][3], sc[n0 + 1][2], sc[n0 + 1][3]);
      pa[kb][2] = pack_e4(sc[n0 + 2][0], sc[n0 + 2][1], sc[n0 + 3][0], sc[n0 + 3][1]);
      pa[kb][3] = pack_e4(sc[n0 + 2][2], sc[n0 + 2][3], sc[n0 + 3][2], sc[n0 + 3][3]);
    }
    const unsigned vt = smem_u32(Vt + (i & 1) * 2 * VTH + h * VTH);
    // fence, both k-steps, commit and wait in one asm: the P fragments and the rescaled O are inputs, so the
    // fence cannot be scheduled ahead of their producers
    WG_PV2(oacc, pa[0], pa[1], desc(vt, 1024, 1), desc(vt + 32, 1024, 1));
    if (i == ntiles - 1) WG_WAIT_SO(sc, oacc);  // last tile's P.V, inside the loop: a wait after the loop (v24's) made
                                                // ptxas serialize every wgmma here (C7515)
  }
  const float l0 = oacc[8][0], l1 = oacc[8][2];  // full row sums (every key column of the tile went through the MMA)
  const float f0 = vsc / (l0 > 0.f ? l0 : 1.f), f1 = vsc / (l1 > 0.f ? l1 : 1.f);
  const long long i0 = ((long long)(b * LQ + j0) * hq + head) * nsplit + s;
  const long long i1 = ((long long)(b * LQ + j1) * hq + head) * nsplit + s;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    *(float2*)(o_part + i0 * D + n * 8 + t4 * 2) = make_float2(oacc[n][0] * f0, oacc[n][1] * f0);
    *(float2*)(o_part + i1 * D + n * 8 + t4 * 2) = make_float2(oacc[n][2] * f1, oacc[n][3] * f1);
  }
  if (t4 == 0) {
    lse_out[i0] = l0 > 0.f ? m0 + log2f(l0) - 8.f : NEG;
    lse_out[i1] = l1 > 0.f ? m1 + log2f(l1) - 8.f : NEG;
  }
}
