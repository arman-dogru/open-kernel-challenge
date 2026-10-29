"""Builds muse_decode_layer/runs/v16_int_splat.py from v14_verify_split.py (exact replacements, each asserted
once): MoE v26a's exact 2-instruction E4M3 scale splat in the decode layer's CUDA expert kernels (moe_mma), with
the same prepare()-time range check and v14's cvt path as fallback. Run at HEAD; the committed file must be
byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v14_verify_split.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''# okc: problem=muse_decode_layer
"""v14:''', '''# okc: problem=muse_decode_layer
"""v16: v14 plus muse_moe_w4a16 v26a's scale splat (check bbebbbdf181f4934, 1.457x there vs v24 1.380x) in the
CUDA expert kernels (moe_mma, small token counts): for a positive normal E4M3 block-scale byte b (0x08..0x7E),
bf16(e4m3(b) * 2^119) is exactly (b << 4) + (239 << 7) per half, so the splat is one PRMT + one IMAD instead of
cvt e4m3x2 -> f16 -> f32, * 2^119, cvt bf16x2. prepare() checks every local scale byte is in 0x08..0x7E and
otherwise compiles v14's cvt path (FASTSCALE 0). Dequantized values are bit-identical to v14's.

v14:''')
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
rep('''def _mod(WARPS, KS, GATED, PF, NB):''', '''FASTSCALE = [0]  # set by _moe_small_bank() from the weights' scale bytes


def _scales_normal(sf, start, count):
    """True if every local block-scale byte is a positive normal E4M3 (0x08..0x7E): the SPLAT fast path is exact."""
    b = sf[start:start + count].contiguous().view(torch.uint8)
    return bool(((b >= 0x08) & (b <= 0x7E)).all())


def _mod(WARPS, KS, GATED, PF, NB):''')
rep('''    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile now: CuPy is lazy (r4 m1)''',
'''    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2
    FASTSCALE[0] = int(_scales_normal(weights["w13_sf"], 0, LOCAL_EXPERTS)
                       and _scales_normal(weights["w2_sf"], 0, LOCAL_EXPERTS))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile now: CuPy is lazy (r4 m1)''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v16_int_splat.py"
dst.write_text(s)
