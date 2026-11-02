"""Builds muse_decode_layer/runs/v20_rows_prefetch.py from v19_nb2_256_evict.py (exact replacements asserted once;
the attention change is tools/rows_prefetch_patch.py, shared with attention v17). Run at HEAD; the committed file must
be byte-identical to its output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rows_prefetch_patch as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
S = [(ROOT / "muse_decode_layer/runs/v19_nb2_256_evict.py").read_text()]


def rep(a, b):
    assert S[0].count(a) == 1, (a[:70], S[0].count(a))
    S[0] = S[0].replace(a, b)


rep('''# okc: problem=muse_decode_layer
"""v19:''', '''# okc: problem=muse_decode_layer
"""v20: v19 with
- full-decode attention at batch >= 2 (split-KV "rows" kernel, 257 us at decode 32 full in v19's profile) on 128-key
  tiles with the next tile's page indices loaded one iteration ahead (PREF): each iteration waits on one DRAM round
  trip (the K / V gather) instead of two (page index, then gather). Same math.
- review of #17: the LAYER_SMALL_MAX comment now says NB 2 from 81 tokens (nearest SMALL_CFG entry, ties to the
  smaller: 80 -> 32, 81 -> 128); the dead NB == 4 launch-bounds branch is removed (no SMALL_CFG entry uses NB 4
  since v18b). v19's profile .log is empty because the kernel prints nothing (BENCH code removed in v18, r10).

v19:''')
rep("LAYER_SMALL_MAX = 256  # tokens up to this use the CUDA experts (v17: <= 32; v19: <= 256, NB 2 above 64)",
    "LAYER_SMALL_MAX = 256  # tokens up to this use the CUDA experts (v17: <= 32; v19: <= 256, NB 2 from 81 tokens)")
rep('''#elif GATED && WARPS == 8 && NB == 4
#define MINB 2
''', '')
rep('''# ~63 -> 30 at 256 on the judge's routing); NB 4 uses split 8 (KS 512) so 32 staged rows fit static shared memory.''',
    '''# ~63 -> 30 at 256 on the judge's routing). NB 4 lost at 256 (v18) and was dropped; v19 runs NB 2 at 256 as well.''')
P.apply(S[0], rep)
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v20_rows_prefetch.py"
dst.write_text(S[0])
