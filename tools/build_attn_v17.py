"""Builds muse_attention_decode/runs/v17_rows_prefetch.py from v15b_verify_route.py (tools/rows_prefetch_patch.py,
exact replacements asserted once). Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rows_prefetch_patch as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
S = [(ROOT / "muse_attention_decode/runs/v15b_verify_route.py").read_text()]


def rep(a, b):
    assert S[0].count(a) == 1, (a[:70], S[0].count(a))
    S[0] = S[0].replace(a, b)


first, rest = S[0].split("\n", 1)
S[0] = first + "\n" + '"""v17: v15b with the full-decode (batch >= 2) split-KV "rows" config on 128-key tiles and the next tile\'s page\nindices loaded one iteration ahead (PREF), as decode v20. Same math and precision.\n\n' + rest[3:]
P.apply(S[0], rep, hw_text="hw")
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_attention_decode/runs/v17_rows_prefetch.py"
dst.write_text(S[0])
