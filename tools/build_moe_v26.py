"""Builds muse_moe_w4a16/runs/v26a_int_splat.py from v24_tuned.py (exact replacements, each asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_moe_w4a16/runs/v24_tuned.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""v24 = v23 with', '''"""v26a = v24 with a 2-instruction E4M3 scale splat (FASTSCALE): for a positive normal E4M3 byte b
(0x08..0x7E), bf16(e4m3(b) * 2^119) is exactly (b << 4) + (239 << 7) (exponent e - 7 + 119 + 127 = e + 239, the
3 mantissa bits land in the top of bf16's 7). The splat is one PRMT (b into both 16-bit halves as 0x00bb00bb)
and one IMAD (* 16 + 0x77807780) instead of v23's cvt e4m3x2 -> f16 -> f32, * 2^119, cvt bf16x2 (~5); the
k-step does 8 splats per 16 x 64 tile, about a third of its instructions (moe_mma is issue-bound: NB = 2's extra
MMAs cost 8-11% in v23's bench). Exhaustively checked on CPU for all 119 bytes. prepare() checks that every
local block-scale byte is in 0x08..0x7E (judge weights: scales clamped to [8, 448], or >= 2^-6 with wide scales,
all normal) and otherwise compiles v24's exact cvt path (FASTSCALE 0). The stored weights are unchanged.

v24 = v23 with''')
rep('''__device__ __forceinline__ u32 scale_splat(u32 b16) {''', '''#if FASTSCALE
// byte k of sw (positive normal E4M3) -> bf16x2 splat of e4m3 * 2^119, exact: (b << 4) + (239 << 7) per half
#define SPLAT(sw, k) (__byte_perm((sw), 0u, 0x4040u + 0x101u * (k)) * 16u + 0x77807780u)
#endif
__device__ __forceinline__ u32 scale_splat(u32 b16) {''')
rep('''    u32 slo = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x22 : 0x00));  // row g
    u32 shi = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x33 : 0x11));  // row g + 8''', '''#if FASTSCALE
    u32 slo = SPLAT(sw, st & 1 ? 2 : 0);  // row g
    u32 shi = SPLAT(sw, st & 1 ? 3 : 1);  // row g + 8
#else
    u32 slo = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x22 : 0x00));  // row g
    u32 shi = scale_splat(__byte_perm(sw, 0, st & 1 ? 0x33 : 0x11));  // row g + 8
#endif''')
rep('''    key = (WARPS, KS, GATED, PF, NB)''', '''    key = (WARPS, KS, GATED, PF, NB, FASTSCALE[0])''')
rep('''f"-DPF={PF}", f"-DNB={NB}"))''', '''f"-DPF={PF}", f"-DNB={NB}", f"-DFASTSCALE={FASTSCALE[0]}"))''')
rep('''def _mod(WARPS, KS, GATED, PF, NB):''', '''FASTSCALE = [0]  # set by prepare() from the weights' scale bytes


def _scales_normal(sf, start, count):
    """True if every local block-scale byte is a positive normal E4M3 (0x08..0x7E): the SPLAT fast path is exact."""
    b = sf[start:start + count].contiguous().view(torch.uint8)
    return bool(((b >= 0x08) & (b <= 0x7E)).all())


def _mod(WARPS, KS, GATED, PF, NB):''')
rep('''    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile the launch configs up front''',
'''    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    FASTSCALE[0] = int(_scales_normal(weights["w13_sf"], start, count) and _scales_normal(weights["w2_sf"], start, count))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile the launch configs up front''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_moe_w4a16/runs/v26a_int_splat.py"
dst.write_text(s)
