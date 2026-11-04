"""Build muse_decode_layer/runs/v26_ldsm.py (decode) and muse_moe_w4a16/runs/v28_ldsm.py (MoE) from decode v25 /
MoE v26e: the CUDA experts (moe_mma) load their B fragments (the staged BF16 activation rows) with
ldmatrix.sync.m8n8.x4 in GEMM 2 (the down projection, not GATED), one per 8-route block per pair of 16-k steps,
issued next to the MMAs: 16 -> 4 shared-load instructions per 64-k tile at NB 2, 8 -> 2 at NB 1. GEMM 1 (GATED) keeps
the scalar loads: there ldmatrix raised spills under its MINB register caps (tools/tests/moe_ldsm_nvrtc.py).
Same values (lane l of matrix j gets row l / 4, columns 2 (l % 4) + {0, 1}: exactly the m16n8k16 B fragment the
scalar loads read), same MMAs in the same order, so outputs are bit-identical. The asm carries the "memory" clobber.
Decode also gets PR #26 review nits n2: the dead NB 4 MINB branch removed, the "NB 4 uses split 8" comment and the
"NB 2 above 64" wording corrected. Run from the repo root."""

OLD = """  const u16* xb = xs + g * XS + 2 * t;
#pragma unroll
  for (int kb = 0; kb < NT; kb++) {
    u32 b0[NB][4], b1[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; nb++)
#pragma unroll
      for (int st = 0; st < 4; st++) {
        b0[nb][st] = *reinterpret_cast<const u32*>(xb + nb * 8 * XS + kb * 64 + st * 16);
        b1[nb][st] = *reinterpret_cast<const u32*>(xb + nb * 8 * XS + kb * 64 + st * 16 + 8);
      }
"""
NEW = """#ifndef LDSM
#define LDSM (!GATED)  // v26 / v28: ldmatrix B fragments in GEMM 2 only (in GATED builds it raised spills under MINB)
#endif
#if LDSM
  // ldmatrix x4: lane l addresses row l % 8 of matrix l / 8; matrix j is k offset 8 j within a pair of 16-k steps,
  // so its registers are b0[st], b1[st], b0[st + 1], b1[st + 1] (row g, k 2t, 2t + 1).
  const unsigned xl = (unsigned)__cvta_generic_to_shared(xs + (lane & 7) * XS + (lane >> 4) * 16 + ((lane >> 3) & 1) * 8);
#else
  const u16* xb = xs + g * XS + 2 * t;
#endif
#pragma unroll
  for (int kb = 0; kb < NT; kb++) {
#if LDSM
    const unsigned xa = xl + 2u * (unsigned)(kb * 64);
#else
    u32 b0[NB][4], b1[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; nb++)
#pragma unroll
      for (int st = 0; st < 4; st++) {
        b0[nb][st] = *reinterpret_cast<const u32*>(xb + nb * 8 * XS + kb * 64 + st * 16);
        b1[nb][st] = *reinterpret_cast<const u32*>(xb + nb * 8 * XS + kb * 64 + st * 16 + 8);
      }
#endif
"""
OLD_TILE = """// grid (N / 16 / WARPS, work bound, K / KS); block 32 * WARPS.
"""
NEW_TILE = """// v26 / v28 (LDSM builds): B fragments by ldmatrix.x4 from the staged rows at shared address xa (this lane's row /
// k offset in tile kb), one x4 per 8-route block per pair of k-steps, loaded next to their MMAs. Same MMAs, same order.
__device__ __forceinline__ void mma_tile_ld(float (*c)[4], uint4 q, uint2 s, unsigned xa) {
  u32 b0[NB][2], b1[NB][2];
#pragma unroll
  for (int st = 0; st < 4; st++) {
    if ((st & 1) == 0) {
#pragma unroll
      for (int nb = 0; nb < NB; nb++)
        asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(b0[nb][0]), "=r"(b1[nb][0]), "=r"(b0[nb][1]), "=r"(b1[nb][1])
            : "r"(xa + 2u * (unsigned)(nb * 8 * XS + st * 16)) : "memory");
    }
    u32 wq = st == 0 ? q.x : st == 1 ? q.y : st == 2 ? q.z : q.w;
    u32 sw = st < 2 ? s.x : s.y;
#if FASTSCALE
    u32 slo = SPLAT(sw, st & 1 ? 2 : 0);  // row g
    u32 shi = SPLAT(sw, st & 1 ? 3 : 1);  // row g + 8
#else
    u32 slo = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x22 : 0x00));  // row g
    u32 shi = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x33 : 0x11));  // row g + 8
#endif
    u32 a0 = bmul2(rotl(wq, 6) & 0x81C081C0u, slo);
    u32 a1 = bmul2(rotl(wq, 2) & 0x81C081C0u, shi);
    u32 a2 = bmul2(rotl(wq, 28) & 0x81C081C0u, slo);
    u32 a3 = bmul2((wq & 0x81808180u) | (rotl(wq, 24) & 0x00400040u), shi);
#pragma unroll
    for (int nb = 0; nb < NB; nb++)
      asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[nb][0]), "+f"(c[nb][1]), "+f"(c[nb][2]), "+f"(c[nb][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[nb][st & 1]), "r"(b1[nb][st & 1]));
  }
}

// grid (N / 16 / WARPS, work bound, K / KS); block 32 * WARPS.
"""
CALLS = [("    mma_tile(cg, q, s, b0, b1);\n", "#if LDSM\n    mma_tile_ld(cg, q, s, xa);\n#else\n    mma_tile(cg, q, s, b0, b1);\n#endif\n")]


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


def kern(text):
    text = rep(text, OLD, NEW)
    text = rep(text, OLD_TILE, NEW_TILE)
    for a, b in CALLS:
        text = rep(text, a, b)
    return text


# ---- decode ----
s = kern(open("muse_decode_layer/runs/v25_ws4_fv.py").read())
s = rep(s, """#elif GATED && WARPS == 8 && NB == 4
#define MINB 2
""", "")
s = rep(s, """# 128 / 256 (v18): NB 2 / NB 4 items (16 / 32 routes per expert chunk) cut the item count (~36 -> 27 at 128 tokens,
# ~63 -> 30 at 256 on the judge's routing); NB 4 uses split 8 (KS 512) so 32 staged rows fit static shared memory.""",
        """# 128 / 256: NB 2 items (16 routes per expert chunk; v18's NB 4 at 256 lost: 115 registers, 2 CTAs/SM).""")
s = rep(s, "(v17: <= 32; v19: <= 256, NB 2 above 64)",
        "(v17: <= 32; v19: <= 256, NB 2 from 81 tokens: nearest entry, ties to the smaller)")
s = rep(s, '# okc: problem=muse_decode_layer\n"""v25:', '''# okc: problem=muse_decode_layer
"""v26: v25 with the CUDA experts' B fragments loaded by ldmatrix.x4 in GEMM 2 (4 instead of 16 shared
loads per 64-k tile at NB 2; bit-identical values and MMAs). tools/build_decode_v26.py.

v25:''')
open("muse_decode_layer/runs/v26_ldsm.py", "w").write(s)
print("wrote muse_decode_layer/runs/v26_ldsm.py")

# ---- MoE ----
m = kern(open("muse_moe_w4a16/runs/v26e_nb2_evict.py").read())
first, rest = m.split("\n", 1)
assert first == "# okc: problem=muse_moe_w4a16" and rest.startswith('"""')
m = (first + '\n"""v28: v26e with moe_mma\'s B fragments loaded by ldmatrix.x4 in GEMM 2 (4 instead of 16 shared\n'
     'loads per 64-k tile at NB 2, 2 instead of 8 at NB 1; bit-identical values and MMAs). tools/build_decode_v26.py.\n\n' + rest[3:])
open("muse_moe_w4a16/runs/v28_ldsm.py", "w").write(m)
print("wrote muse_moe_w4a16/runs/v28_ldsm.py")
