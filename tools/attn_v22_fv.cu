// Full-attention verify (q_len 8) split-KV partials for tp8 (16 query / 2 KV heads, D 64), attention v22.
// One CTA per (split s, request b), 8 warps: warps 0-3 serve KV head 0, 4-7 KV head 1; each warp owns 16 rows
// (row r = token * 8 + query head within the group, as the Triton head kernel). K/V rows (both heads, 128 B per
// key) are gathered through the page table by a 3-stage cp.async pipeline of 64-key tiles (16-byte chunks
// XOR-swizzled by key & 7), page indices loaded one tile ahead. S = Q.K^T on FP8 mma.sync m16n8k32 (queries
// already E4M3, as FA3's FP8 path), online softmax in registers (exp2 domain), P (FP16) . V on mma.sync
// m16n8k16 with V converted E4M3 -> FP16 (exact) into a swizzled tile read by ldmatrix.trans. Output: per row
// the partial o = acc * v_scale / l and lse = m + log2(l) in the Triton head kernel's layout, so the same
// combine/norm/gate kernel merges them.
#define D 64
#define HKV 2
#define G 8
#define LQ 8
#define BN 64
#define ST 3
#define KROW 128
#define VROW 256
#define TILE (BN * KROW)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  unsigned r;
  asm volatile("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
  unsigned r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ unsigned e4m3x2_h2(unsigned short x) {
  unsigned r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(x)); return r;
}

extern "C" __global__ void __launch_bounds__(256, 2)
attn_fv(const unsigned char* __restrict__ q, int stride_qt, const unsigned char* __restrict__ kc,
        const unsigned char* __restrict__ vc, const int* __restrict__ pt, int stride_pt,
        const int* __restrict__ seqlens, const float* __restrict__ ksp, const float* __restrict__ vsp,
        float* __restrict__ o_part, float* __restrict__ lse_out, float qk_scale, int chunk, int page,
        int nsplit, int hq) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* Ks = smem;
  unsigned char* Vs = smem + ST * TILE;
  unsigned char* V16 = smem + 2 * ST * TILE;
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

  unsigned qa[2][4];
  {
    const unsigned char* q0 = q + (long long)(b * LQ + j0) * stride_qt + head * D;
    const unsigned char* q1 = q + (long long)(b * LQ + j1) * stride_qt + head * D;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      qa[ks][0] = *(const unsigned*)(q0 + ks * 32 + t4 * 4);
      qa[ks][1] = *(const unsigned*)(q1 + ks * 32 + t4 * 4);
      qa[ks][2] = *(const unsigned*)(q0 + ks * 32 + 16 + t4 * 4);
      qa[ks][3] = *(const unsigned*)(q1 + ks * 32 + 16 + t4 * 4);
    }
  }
  const float scale = qk_scale * ksp[0] * 1.4426950408889634f;
  const float vsc = vsp[0];

  // loader: key kl of the tile, 16-byte chunks c0, c0 + 1 of its 128-byte K and V rows
  const int kl = tid >> 2, c0 = (tid & 3) * 2;
  auto rowof = [&](int tile) -> long long {
    int t = lo + tile * BN + kl;
    if (tile < ntiles && t < hi) {
      int p = pt[(long long)b * stride_pt + t / page];
      return (long long)p * page + t % page;
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

  float oacc[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
  float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;

  issue(0, rowof(0));
  issue(1, rowof(1));
  long long nextrow = rowof(2);
  for (int i = 0; i < ntiles; ++i) {
    asm volatile("cp.async.wait_group 1;" ::);
    __syncthreads();
    {  // V tile i: E4M3 -> FP16 (exact) into V16 [key][128 halves], 16-byte chunks swizzled by key & 7
      const unsigned char* vsrc = Vs + (i % ST) * TILE + kl * KROW;
#pragma unroll
      for (int c = c0; c < c0 + 2; ++c) {
        uint4 w = *(const uint4*)(vsrc + ((c ^ (kl & 7)) * 16));
        unsigned u[4] = {w.x, w.y, w.z, w.w};
        unsigned h2[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          h2[2 * k] = e4m3x2_h2((unsigned short)(u[k] & 0xffff));
          h2[2 * k + 1] = e4m3x2_h2((unsigned short)(u[k] >> 16));
        }
        unsigned char* vrow = V16 + kl * VROW;
        *(uint4*)(vrow + (((2 * c) ^ (kl & 7)) * 16)) = make_uint4(h2[0], h2[1], h2[2], h2[3]);
        *(uint4*)(vrow + (((2 * c + 1) ^ (kl & 7)) * 16)) = make_uint4(h2[4], h2[5], h2[6], h2[7]);
      }
    }
    issue(i + 2, nextrow);
    nextrow = rowof(i + 3);
    __syncthreads();

    const unsigned char* Kt = Ks + (i % ST) * TILE;
    float sc[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
      const unsigned char* krow = Kt + (n * 8 + g) * KROW;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int cc = h * 4 + ks * 2;
        unsigned b0 = *(const unsigned*)(krow + ((cc ^ g) * 16) + t4 * 4);
        unsigned b1 = *(const unsigned*)(krow + (((cc + 1) ^ g) * 16) + t4 * 4);
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(sc[n][0]), "+f"(sc[n][1]), "+f"(sc[n][2]), "+f"(sc[n][3])
                     : "r"(qa[ks][0]), "r"(qa[ks][1]), "r"(qa[ks][2]), "r"(qa[ks][3]), "r"(b0), "r"(b1));
      }
    }
    const int tb = lo + i * BN + t4 * 2;
    float mx0 = NEG, mx1 = NEG;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int t = tb + n * 8 + e;
        const float v0 = (t < hi && t <= pos0) ? sc[n][e] * scale : NEG;
        const float v1 = (t < hi && t <= pos1) ? sc[n][2 + e] * scale : NEG;
        sc[n][e] = v0; sc[n][2 + e] = v1;
        mx0 = fmaxf(mx0, v0); mx1 = fmaxf(mx1, v1);
      }
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float ms0 = mn0 == NEG ? 0.f : mn0, ms1 = mn1 == NEG ? 0.f : mn1;
    const float a0 = ex2(m0 - ms0), a1 = ex2(m1 - ms1);
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sc[n][0] = ex2(sc[n][0] - ms0); sc[n][1] = ex2(sc[n][1] - ms0);
      sc[n][2] = ex2(sc[n][2] - ms1); sc[n][3] = ex2(sc[n][3] - ms1);
      s0 += sc[n][0] + sc[n][1]; s1 += sc[n][2] + sc[n][3];
      oacc[n][0] *= a0; oacc[n][1] *= a0; oacc[n][2] *= a1; oacc[n][3] *= a1;
    }
    l0 = l0 * a0 + s0; l1 = l1 * a1 + s1;
    m0 = mn0; m1 = mn1;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const unsigned pa0 = pack_h2(sc[2 * kk][0], sc[2 * kk][1]);
      const unsigned pa1 = pack_h2(sc[2 * kk][2], sc[2 * kk][3]);
      const unsigned pa2 = pack_h2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]);
      const unsigned pa3 = pack_h2(sc[2 * kk + 1][2], sc[2 * kk + 1][3]);
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int mi = lane >> 3, rr = lane & 7;
        const int key = kk * 16 + (mi & 1) * 8 + rr;
        const int chunkv = h * 8 + np * 2 + (mi >> 1);
        const unsigned addr = smem_u32(V16 + key * VROW + ((chunkv ^ (key & 7)) * 16));
        unsigned r0, r1, r2, r3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(oacc[2 * np][0]), "+f"(oacc[2 * np][1]), "+f"(oacc[2 * np][2]), "+f"(oacc[2 * np][3])
                     : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(r0), "r"(r1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(oacc[2 * np + 1][0]), "+f"(oacc[2 * np + 1][1]), "+f"(oacc[2 * np + 1][2]),
                       "+f"(oacc[2 * np + 1][3])
                     : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(r2), "r"(r3));
      }
    }
  }
  asm volatile("cp.async.wait_group 0;" ::);
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float f0 = vsc / (l0 > 0.f ? l0 : 1.f), f1 = vsc / (l1 > 0.f ? l1 : 1.f);
  const long long i0 = ((long long)(b * LQ + j0) * hq + head) * nsplit + s;
  const long long i1 = ((long long)(b * LQ + j1) * hq + head) * nsplit + s;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    *(float2*)(o_part + i0 * D + n * 8 + t4 * 2) = make_float2(oacc[n][0] * f0, oacc[n][1] * f0);
    *(float2*)(o_part + i1 * D + n * 8 + t4 * 2) = make_float2(oacc[n][2] * f1, oacc[n][3] * f1);
  }
  if (t4 == 0) {
    lse_out[i0] = l0 > 0.f ? m0 + log2f(l0) : NEG;
    lse_out[i1] = l1 > 0.f ? m1 + log2f(l1) : NEG;
  }
}
