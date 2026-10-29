"""Build muse_bf16_proj/runs/v29b_route_m8.py from v29_wgmma_partials.py (byte-exact, rerunnable).

v29b = v29 with WG_ROUTE = {("pooler", 8), ("qkv_gate", 8)}: the cheapest single measurement of v29's deterministic
partial stores + last-CTA reduction (STATUS, Oct 29 ~4:30 AM). Only m = 8 (verify 1) pooler and qkv_gate take the
wgmma path; every other (layer, m) runs v17's routes as v29. Projected ~20.5 vs v17 20.3 us (pooler) and ~14.7 vs
14.0 us (qkv_gate): a measurement of the reduction's real cost, not an expected board win.
"""
import sys
SRC, DST = "muse_bf16_proj/runs/v29_wgmma_partials.py", "muse_bf16_proj/runs/v29b_route_m8.py"
s = open(SRC).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


sub("WG_ROUTE = frozenset()\n", 'WG_ROUTE = frozenset({("pooler", 8), ("qkv_gate", 8)})  # v29b: measure the reduction at m 8\n')
first = s.index('"""')
s = s[:first] + '"""v29b: v29 with WG_ROUTE {("pooler", 8), ("qkv_gate", 8)} (see tools/build_bf16_v29b.py).\n\n' + s[first + 3:]
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
