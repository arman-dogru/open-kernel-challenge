"""Build muse_attention_decode/runs/v24_rows_occ4.py from runs/v23_fv_addr.py (board 1.478x). Run from the repo root.

v24 = v23 with full-attention decode (q_len 1) at batch >= 2 on the split-KV "rows" kernel at ~4 programs per SM
(v17-v23: ~16; same 128-key tiles, V hoisted, page-index prefetch). Evidence: decode-layer diagnostic checks
6567d26179874f80 / 8bae0dfe527f4f01 (in-prepare sweeps on the judge's H200, cold L2, ctx 80000, page 1, agentic):
this config 127.6 / 126.4 us at batch 16 vs 150.6 us for v23's, 234.0-234.7 us at batch 32 vs 246.9 us; decode v24
(same change) measured decode 16 / 32 full 311 / 450 -> 292 / 441 us in the decode layer (check 8bae0dfe527f4f01).
"""
import sys

SRC = "muse_attention_decode/runs/v23_fv_addr.py"
DST = "muse_attention_decode/runs/v24_rows_occ4.py"
s = open(SRC).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


s = rep(s, '    return ("rows", 128, 4, 3, 16, True)  # 128-key tiles + page-index prefetch (PREF = hoist), v20 / v17',
        '    return ("rows", 128, 4, 3, 4, True)  # v24: ~4 programs/SM (was 16); 128-key tiles + PREF (= hoist), v20 / v17')
s = rep(s, '''#   full decode, batch >= 2                  "rows", ~16 programs/SM, V load hoisted: 1.17-1.26x
#                                            at b 16/32 (the 4/SM "head" config was 0.92x there)''',
        '''#   full decode, batch >= 2                  "rows", 128-key tiles, ~4 programs/SM (v24; ~16 before: 150.6 /
#                                            246.9 vs 126-128 / 234 us at b 16/32, decode checks 6567d261 / 8bae0dfe)''')
s = rep(s, '# okc: problem=muse_attention_decode\n"""v23:', '''# okc: problem=muse_attention_decode
"""v24: v23 with full-attention decode (q_len 1) at batch >= 2 on the split-KV "rows" kernel at ~4 programs per SM
instead of ~16 (same 128-key tiles, V hoisted, page-index prefetch): fewer, longer splits. In-prepare sweeps on the
judge's H200 (decode-layer diagnostic checks 6567d26179874f80 / 8bae0dfe527f4f01, cold L2): rows + combine
150.6 -> ~127 us at batch 16 and 246.9 -> ~234 us at batch 32 (tools/build_attn_v24.py).

v23:''')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
