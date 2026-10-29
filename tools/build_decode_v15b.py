"""Builds muse_decode_layer/runs/v15b_moe32.py from v14_verify_split.py (exact replacements, each asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v14_verify_split.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""v14: v13 plus', '''"""v15b: v14 with the experts re-routed from v15a's in-prepare bench (check 75b0321a5df9459d, synthetic
heavy-tailed routings): the CUDA tensor-core experts now also at 17-32 tokens with (NB 1, 16 warps, split 4)
(160 us vs Marlin's 178.5 us at 32 tokens / 49 local routes; 8 warps 160.4 us), and Marlin moe_block_size by
token range, 16 at 65-383 tokens (256 tokens: 441 us at 16 vs 459 us at 32). Everything else is v14's.

v14: v13 plus''')
rep('''# The layer uses these CUDA experts up to LAYER_SMALL_MAX = 16 tokens only, so v24's 32-token entry is not carried.
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2))}''',
    '''# 32: from the layer's own bench (v15a, check 75b0321a5df9459d).
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (16, 4, 2), (16, 4, 2))}''')
rep("LAYER_SMALL_MAX = 16  # tokens up to this use the CUDA experts (v9 judge numbers)",
    "LAYER_SMALL_MAX = 32  # tokens up to this use the CUDA experts (v9 judge numbers to 16, v15a bench at 32)")
rep('''# Marlin moe_block_size per token count, measured on muse_moe_w4a16 (SGLang picks 48 / 64)
MOE_BLOCK_M_OVERRIDE = {128: 16, 256: 32}''',
    '''# Marlin moe_block_size by token range (lo, hi, block): 16 measured best at 128 (muse_moe_w4a16) and at 256
# in the layer's bench (v15a); SGLang picks 48 / 64 there.
MOE_BLOCK_M_OVERRIDE = ((65, 383, 16),)''')
rep("    block_size_m = MOE_BLOCK_M_OVERRIDE.get(num_tokens, block_size_m)\n",
    "    for lo, hi, bm in MOE_BLOCK_M_OVERRIDE:\n        if lo <= num_tokens <= hi:\n            block_size_m = bm\n")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v15b_moe32.py"
out.write_text(s)
