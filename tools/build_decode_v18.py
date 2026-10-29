"""Builds muse_decode_layer/runs/v18_cuda_experts_256.py from v17_fused_chain.py (exact replacements, each asserted
once). Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v17_fused_chain.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''# okc: problem=muse_decode_layer
"""v17:''', '''# okc: problem=muse_decode_layer
"""v18: v17 with the CUDA tensor-core experts extended from <= 32 to <= 256 tokens (verify 16 / 32: 128 / 256 tokens,
which ran Marlin): NB 2 work items (16 routes per expert chunk) at 128 tokens and NB 4 (32 routes, split 8 so the
staged rows fit 48 KB of static shared memory) at 256. Projection (judge routing, 100 seeds; v26c's measured
bs 8/16/32 fit time = 25.5 + 7.86 us per 8-route item; NB 2 / 4 items cost ~1.10 / ~1.35-1.45x, v23's bench):
~259 us at 128 tokens and ~343-366 us at 256 vs Marlin's ~339 / ~469 us standalone. GATED occupancy caps scale with NB
(NB 2: 3 CTAs/SM, NB 4: 2). FASTSCALE now lives in the MoE state (r10 minor) instead of a module global. More than 256
tokens still run Marlin.

v17:''')
rep('''#if GATED && WARPS == 8
#define MINB 4
#elif GATED && WARPS == 4
#define MINB 7
#else''', '''#if GATED && WARPS == 8 && NB == 1
#define MINB 4
#elif GATED && WARPS == 4 && NB == 1
#define MINB 7
#elif GATED && WARPS == 8 && NB == 2
#define MINB 3
#elif GATED && WARPS == 8 && NB == 4
#define MINB 2
#else''')
rep('''FASTSCALE = [0]  # set by _moe_small_bank() from the weights' scale bytes


''', '')
rep('''def _mod(WARPS, KS, GATED, PF, NB):''', '''def _mod(WARPS, KS, GATED, PF, NB, FAST):''')
rep('''    key = (WARPS, KS, GATED, PF, NB, FASTSCALE[0])''', '''    key = (WARPS, KS, GATED, PF, NB, FAST)''')
rep('''f"-DFASTSCALE={FASTSCALE[0]}"))''', '''f"-DFASTSCALE={FAST}"))''')
rep('''    m1 = _mod(W1, H // S1, 1, P1, NB)
    m2 = _mod(W2, I // S2, 0, P2, NB)''', '''    m1 = _mod(W1, H // S1, 1, P1, NB, state["fast"])
    m2 = _mod(W2, I // S2, 0, P2, NB, state["fast"])''')
rep('''    FASTSCALE[0] = int(_scales_normal(weights["w13_sf"], 0, LOCAL_EXPERTS)
                       and _scales_normal(weights["w2_sf"], 0, LOCAL_EXPERTS))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile now: CuPy is lazy (r4 m1)
        m1 = _mod(W1, H // S1, 1, P1, NB)
        for name in ("route_prep", "moe_mma"):
            m1.get_function(name)
        _mod(W2, I // S2, 0, P2, NB).get_function("moe_mma")''', '''    # FASTSCALE per state (r10): the PRMT + IMAD splat only when every local scale byte is a positive normal E4M3
    st["fast"] = fast = int(_scales_normal(weights["w13_sf"], 0, LOCAL_EXPERTS)
                            and _scales_normal(weights["w2_sf"], 0, LOCAL_EXPERTS))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile now: CuPy is lazy (r4 m1)
        m1 = _mod(W1, H // S1, 1, P1, NB, fast)
        for name in ("route_prep", "moe_mma"):
            m1.get_function(name)
        _mod(W2, I // S2, 0, P2, NB, fast).get_function("moe_mma")''')
rep('''SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2))}''', '''# 128 / 256 (v18): NB 2 / NB 4 items (16 / 32 routes per expert chunk) cut the item count (~36 -> 27 at 128 tokens,
# ~63 -> 30 at 256 on the judge's routing); NB 4 uses split 8 (KS 512) so 32 staged rows fit static shared memory.
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2)), 256: (4, (8, 8, 2), (8, 8, 2))}''')
rep("LAYER_SMALL_MAX = 32  # tokens up to this use the CUDA experts (v9 judge numbers to 16; v17: 17-32, see docstring)",
    "LAYER_SMALL_MAX = 256  # tokens up to this use the CUDA experts (v17: <= 32; v18: <= 256, see docstring)")
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v18_cuda_experts_256.py"
dst.write_text(s)
