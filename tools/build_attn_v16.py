"""Builds muse_attention_decode/runs/v16<tag>.py from v15b_verify_route.py with the full-verify (q_len 8, batch >= 2)
route given on the command line. Exact replacements, each asserted once; run at HEAD and compare byte-for-byte.
usage: python3 tools/build_attn_v16.py <out.py> <doc line> <small-batch path> <large-batch path> <split batch>
  a path is "kernel,BLOCK_N,warps,stages,occ,hoist" (hoist 0/1) or "fa3"."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_attention_decode/runs/v15b_verify_route.py").read_text()
out, doc, pa, pb, split = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


def lit(p):
    if p == "fa3":
        return "None"
    k, bn, w, st, occ, h = p.split(",")
    return f'("{k}", {int(bn)}, {int(w)}, {int(st)}, {int(occ)}, {bool(int(h))})'


rep('"""v15b: v14b with', f'"""{doc}\n\nv15b: v14b with')
rep("like 132 measured slower than 128 at b16, check 85e63326). The combine masks either way.",
    "like 132 measured slower than 128 at b16, check 85e63326). The combine needs a power of two.")
rep("#   sliding verify (q_len 8), any batch      split-KV \"head\" kernel (FP8 Q.K): 1.15-1.22x FA3",
    "#   sliding verify (q_len 8), any batch      split-KV \"head\" kernel (FP8 Q.K): 1.11-1.22x FA3")
rep('''        if kind == "full" and batch >= 2:
            return None  # FA3 (split-KV head kernel 0.67-0.69x of it at b16/32, check 28ad328cf5284948)''',
    f'''        if kind == "full" and batch >= 2:  # v16: see the docstring
            return {lit(pa)} if batch <= {split} else {lit(pb)}''')
Path(out).write_text(s)
