// Full-attention verify (q_len 8) split-KV partials for tp8 (16 query / 2 KV heads, D 64), attention v22.
// One CTA per (split s, request b), 8 warps: warps 0-3 serve KV head 0, 4-7 KV head 1; each warp owns 16 rows
// (row r = token * 8 + query head within the group, as the Triton head kernel). K/V rows (both heads, 128 B per
// key) are gathered through the page table by an ST-stage cp.async pipeline of 64-key tiles (16-byte chunks
// XOR-swizzled by key & 7), page indices loaded one tile ahead. S = Q.K^T on FP8 mma.sync m16n8k32 (queries
// already E4M3, as FA3's FP8 path) on wgmma m64n64k32 (one warpgroup per KV head: its 64 rows = 8 query heads x 8
// tokens; A = Q fragments from registers, B = the K tile through a 128-byte-swizzle descriptor), online softmax in
// registers (exp2 domain), P . V also on FP8 wgmma (A = P from registers, B = V^T in 128-byte-swizzle rows): P (scaled by 2^8 into E4M3's range) is packed straight from the S accumulators, and V is transposed per
// tile into V^T [head * 64 + d][key slot] with key slots permuted to match the S accumulator layout (4x4 byte
// transposes by prmt), so a B fragment is one 32-bit shared load. Output: per row
// the partial o = acc * v_scale / l and lse = m + log2(l) in the Triton head kernel's layout, so the same
// combine/norm/gate kernel merges them.
#define D 64
#define HKV 2
#define G 8
#define LQ 8
#define BN 64
#define ST 3  // cp.async stages: ST - 1 tiles (16 KB each) in flight per CTA
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
#define WG_QK(acc, a, d, sc)                                                                                        \
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %38, 0;\n"                                                         \
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13," \
               "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, "   \
               "p, 1, 1; }"                                                                                         \
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]),              \
                 "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]),              \
                 "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]),              \
                 "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]),              \
                 "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]),              \
                 "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]),              \
                 "+f"(acc[7][2]), "+f"(acc[7][3])                                                                 \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(d), "n"(0), "r"(sc))
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
__device__ __forceinline__ void wg_commit_wait() {
  asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;" ::: "memory");
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned pack_e4(float x0, float x1, float x2, float x3) {  // bytes x0 (low) .. x3
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (unsigned)lo | ((unsigned)hi << 16);
}

extern "C" __global__ void __launch_bounds__(256, 2)
attn_fv(const unsigned char* __restrict__ q, int stride_qt, const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc, const int* __restrict__ pt, int stride_pt,
        const int* __restrict__ seqlens, const float* __restrict__ ksp, const float* __restrict__ vsp,
        float* __restrict__ o_part, float* __restrict__ lse_out, float qk_scale, int chunk, int page_shift,
        int nsplit, int hq) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);  // 1024-aligned (SW128 atoms)
  unsigned char* Ks = smem;
  unsigned char* Vs = smem + ST * TILE;
  unsigned char* Vt = smem + 2 * ST * TILE;
  unsigned char* Qs = Vt + 4 * VTH;  // V^T is double-buffered (P.V of tile i runs into tile i + 1)
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

  {  // Q -> shared memory once (wgmma A operand by descriptor). Row m of KV head hh's 64 = token 2 (m >> 4) +
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

  // loader: key kl of the tile, 16-byte chunks c0, c0 + 1 of its 128-byte K and V rows
  const int kl = tid >> 2, c0 = (tid & 3) * 2;
  auto rowof = [&](int tile) -> long long {
    int t = lo + tile * BN + kl;
    if (tile < ntiles && t < hi) {
      int p = pt[(long long)b * stride_pt + (t >> page_shift)];
      return ((long long)p << page_shift) + (t & ((1 << page_shift) - 1));
    }
    return -1;
  };
  auto issue = [&](int tile, long long row) {
    if (tile < ntiles) {
      unsigned kd = smem_u32(Ks + (tile % ST) * TILE + kl * KROW);
      unsigned vd = smem_u32(Vs + (tile % ST) * TILE + kl * KROW);
      const int n = row >= 0 ? 16 : 0;
      const long long r = row >= 0 ? row : 0;
#pragma unroll
      for (int c = c0; c < c0 + 2; ++c) {
        const int ph = ((c ^ (kl & 7)) * 16);
        cp16(kd + ph, kc + r * KROW + c * 16, n);
        cp16(vd + ph, vc + r * KROW + c * 16, n);
      }
    }
    asm volatile("cp.async.commit_group;" ::);
  };

  float oacc[9][4];  // [0..7]: O (64 dims); [8]: sum of P over keys (V^T rows 64..71 are ones)
#pragma unroll
  for (int n = 0; n < 9; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
  float m0 = NEG, m1 = NEG;
  for (int k = tid; k < 2 * 2 * 8 * 8; k += 256) {  // ones rows: 2 buffers x 2 heads x 8 rows x 8 chunks (E4M3 1.0)
    const int buf_head = k >> 6, rr = (k >> 3) & 7, c = k & 7;
    *(uint4*)(Vt + buf_head * VTH + (64 + rr) * 128 + c * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }

  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // Q (st.shared) -> wgmma; the loop's first barrier orders it
#pragma unroll
  for (int k = 0; k < ST - 1; ++k) issue(k, rowof(k));
  long long nextrow = rowof(ST - 1);
  for (int i = 0; i < ntiles; ++i) {
    asm volatile("cp.async.wait_group %0;" :: "n"(ST - 2));
    __syncthreads();
    {  // V tile i -> V^T: task = 4 dims (dg) x 4 key slots (sg); slot 4t + 2a' + e of 32-key block <- key 8(2A + a') + 2t + e
      const unsigned char* vsrc = Vs + (i % ST) * TILE;
#pragma unroll
      for (int rep = 0; rep < 2; ++rep) {
        const int combo = warp + 8 * rep;
        const int dg = (lane & 7) + 8 * (combo & 3), sg = (lane >> 3) + 4 * (combo >> 2);
        const int blk = sg >> 3, sl = sg & 7, tt = sl & 3, A = sl >> 2;
        const int k0 = blk * 32 + 16 * A + 2 * tt;
        const int kk[4] = {k0, k0 + 1, k0 + 8, k0 + 9};
        unsigned w[4];
#pragma unroll
        for (int r = 0; r < 4; ++r)
          w[r] = *(const unsigned*)(vsrc + kk[r] * KROW + ((((dg >> 2) ^ (kk[r] & 7)) * 16) + (dg & 3) * 4));
        const unsigned x = __byte_perm(w[0], w[1], 0x5140), y = __byte_perm(w[2], w[3], 0x5140);
        const unsigned x2 = __byte_perm(w[0], w[1], 0x7362), y2 = __byte_perm(w[2], w[3], 0x7362);
        const int slot = blk * 32 + sl * 4;
        const unsigned o4[4] = {__byte_perm(x, y, 0x5410), __byte_perm(x, y, 0x7632), __byte_perm(x2, y2, 0x5410),
                                __byte_perm(x2, y2, 0x7632)};
#pragma unroll
        for (int r = 0; r < 4; ++r) {  // row = head * 64 + d; 128-byte rows, 16-byte chunk ^= row & 7
          const int row = dg * 4 + r;
          *(unsigned*)(Vt + (i & 1) * 2 * VTH + row * 128 + (((slot >> 4) ^ (row & 7)) << 4) + (slot & 15)) = o4[r];
        }
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // cp.async / st.shared -> wgmma (async proxy)
    __syncthreads();
    issue(i + ST - 1, nextrow);
    nextrow = rowof(i + ST);

    const unsigned kt = smem_u32(Ks + (i % ST) * TILE) + h * 64;
    float sc[8][4] = {};
    wg_fence();
    const unsigned qs = smem_u32(Qs + h * QH);
    WG_QK_SS(sc, desc(qs, 1024, 1), desc(kt, 1024, 1), 0);
    WG_QK_SS(sc, desc(qs + 32, 1024, 1), desc(kt + 32, 1024, 1), 1);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    WG_WAIT_SO(sc, oacc);  // S(i) and the previous tile's P.V (overlapped with this tile's barrier/transpose/QK)
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
  }
  {
    float dummy[8][4] = {};
    WG_WAIT_SO(dummy, oacc);  // last tile's P.V
  }
  asm volatile("cp.async.wait_group 0;" ::);
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
