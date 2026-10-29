"""Builds muse_decode_layer/runs/v14_verify_split.py from v13_fused_attn.py (exact string replacements, each
asserted to match once). Run at HEAD: python3 tools/build_decode_v14.py [out]; the committed file must be
byte-identical to its output (r7 m2)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v13_fused_attn.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''"""v13: v12 plus''', '''"""v14: v13 plus muse_attention_decode v15b's verify routing (check ab5f387ae32242ba, 1.376x there): 8-token
verify blocks run the split-KV "head" kernel (FP8 x FP8 Q.K, BF16 P.V, FP32 softmax) on sliding layers at any
batch and on full layers at batch 1; full-layer verify at batch >= 2 stays on FA3 (the head kernel was 0.67-0.69x
of it there). Cleanups: the combine's dead NSPLIT_P2 masking removed, with a power-of-two assert in _nsplit (r6),
the _nsplit docstring (r6 n3), the layer summary / "Everything else" line now name the fused q/k front end
(r7 n1), and the RoPE clamp comment argues from the sliding bound (r7 n2). Precision is unchanged.

v13: v12 plus''')
rep('''  attention       split-KV Triton decode kernel + combine fused with the per-head norm x gate
                  (muse_attention_decode), on decode steps by regime (_attn_path); FA3 on verify''',
    '''  q/k front end   one fused Triton norm + RoPE + FP8 q + K/V cache-write program (muse_attention_decode
                  v13/v14) instead of FusedQKNorm, the casts and store_cache, on every path
  attention       split-KV Triton decode kernel + combine fused with the per-head norm x gate
                  (muse_attention_decode), by regime (_attn_path): decode steps, sliding verify and
                  full verify at batch 1; FA3 for full-layer verify at batch >= 2''')
rep("Everything else (norms, routing, Marlin experts, unpooler) is SGLang's, unchanged.",
    "Everything else (the layer's other norms, routing, Marlin experts above 16 tokens, unpooler) is SGLang's,\nunchanged.")
rep("            # r6 n4: clamp to the table (2^21 rows, as SGLang's); judge positions stay < 2^20 + 9\n",
    "            # r6 n4: clamp to the table (2^21 rows, as SGLang's). RoPE runs on sliding layers only, whose judge\n"
    "            # positions stay <= ~300,007, so the clamp never changes a judge input.\n")
rep('''                       HQ: tl.constexpr, NSPLIT: tl.constexpr, D: tl.constexpr, NSPLIT_P2: tl.constexpr):''',
    '''                       HQ: tl.constexpr, NSPLIT: tl.constexpr, D: tl.constexpr):''')
rep("    NSPLIT need not be a power of two: NSPLIT_P2 = next power of two, extra lanes masked.\"\"\"",
    "    NSPLIT is a power of two (_nsplit asserts it), so no lanes are masked.\"\"\"")
rep("    sp = tl.arange(0, NSPLIT_P2)\n    spm = sp < NSPLIT\n", "    sp = tl.arange(0, NSPLIT)\n")
rep('lse = tl.load(lse_ptr + idx, mask=spm, other=float("-inf"))', "lse = tl.load(lse_ptr + idx)")
rep("o = tl.load(o_ptr + idx[:, None] * D + d[None, :], mask=spm[:, None], other=0.0)",
    "o = tl.load(o_ptr + idx[:, None] * D + d[None, :])")
rep("NSPLIT=nsplit, D=d, NSPLIT_P2=triton.next_power_of_2(nsplit),\n", "NSPLIT=nsplit, D=d,\n")
rep('''    """Splits per request: ~occ programs per SM, slices of about 256 keys. Rounding to the nearest
    power of two can go up, so slices can be shorter than 256 keys (capacity 1800 -> 8 splits of 225)."""''',
    '''    """Splits per request, a power of two in 1-256: ~occ programs per SM and at most ~span / 256 splits before
    rounding. Rounding to the nearest power of two can go up, so slices can be shorter than 256 keys (capacity
    1800 -> 8 splits of 225), never below ~128."""''')
rep("like 132 measured slower than 128 at b16, check 85e63326). The combine masks either way.",
    "like 132 measured slower than 128 at b16, check 85e63326). The combine relies on a power of two.")
rep("    return min(2 * p, 256) if 2 * p - n < n - p else p\n",
    "    r = min(2 * p, 256) if 2 * p - n < n - p else p\n    assert r & (r - 1) == 0\n    return r\n")
rep("# Attention path: muse_attention_decode's _attn_path (v9_regime), verbatim.",
    "# Attention path: muse_attention_decode v15b's _attn_path, verbatim.")
rep('''#   any verify block (q_len 8)               SGLang's FA3: our 64-128-row blocks lose to it
#                                            on full layers (0.35-0.8x) and only tie or win by
#                                            <8% on sliding ones, so FA3 is kept at every batch''',
    '''#   sliding verify (q_len 8), any batch      split-KV "head" kernel (FP8 Q.K): 1.15-1.22x FA3
#   full verify, batch 1                     split-KV "head" kernel: 1.36x FA3
#   full verify, batch >= 2                  SGLang's FA3 (the head kernel was 0.67-0.69x at b16/32;
#                                            attention checks 28ad328cf5284948 / ab5f387ae32242ba)''')
rep('''def _attn_path(kind, L, batch, hkv, d):
    if L != 1:
        return None
''', '''def _attn_path(kind, L, batch, hkv, d):
    if L != 1:
        if hkv * d > 128 or (kind == "full" and batch >= 2):
            return None
        return ("head", 64, 4, 3, 4, False)
''')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v14_verify_split.py"
out.write_text(s)
