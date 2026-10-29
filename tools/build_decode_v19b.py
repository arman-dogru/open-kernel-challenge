"""Builds muse_decode_layer/runs/v19b_nb2_minb2.py from v19_nb2_256_evict.py (exact replacements, each asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v19_nb2_256_evict.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''# okc: problem=muse_decode_layer
"""v19:''', '''# okc: problem=muse_decode_layer
"""v19b: v19 with the NB 2 gated expert kernel (GEMM-1 at 128 / 256 tokens) capped for 2 CTAs per SM instead of 3:
NVRTC gives 88 registers and no spill (v19: 80 registers, a 24-byte spill with FASTSCALE). A register / spill vs
occupancy test; everything else is v19's.

v19:''')
rep('''#elif GATED && WARPS == 8 && NB == 2
#define MINB 3''', '''#elif GATED && WARPS == 8 && NB == 2
#define MINB 2''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v19b_nb2_minb2.py"
dst.write_text(s)
