"""Build muse_attention_decode/runs/v19_tok_norm.py from v17_rows_prefetch.py (byte-exact, rerunnable).

v19 = v17 + two launch-shape changes on the sliding-verify-dominant costs (v17 profile 8deec285efbe4b57: verify 32
sliding = split 16.2 + norm/RoPE/store 4.4 + combine 4.0 us; verify 16 = 10.0 + 2.8 + 2.7):
  A. _norm_rope_store_kernel: one program per token over all HQ + 2*HKV heads as a [32, D/2] tile (2 warps) instead of
     one 1-warp program per (token, head) (5,120 programs at T = 256). Same per-head math (FP32 RMSNorm, RoPE, BF16
     round of K, FP8 casts); every cell runs it, FA3 ones included.
  B. Sliding verify: ~2 head-kernel programs per SM (occ 2) instead of 4: at batch 32 NSPLIT 8 -> 4 (512 programs in
     ~2 waves of 5 tiles -> 256 in one wave of 9) and half the FP32 partials the combine reads. batch 1/16 keep 8.
"""
import sys
SRC, DST = "muse_attention_decode/runs/v17_rows_prefetch.py", "muse_attention_decode/runs/v19_tok_norm.py"
s = open(SRC).read()


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


a = s.index("def _norm_rope_store_kernel(")
b = s.index("def _norm_rope_store_fused(")
sub(s[a:b], '''def _norm_rope_store_kernel(qkv_ptr, stride_qkv, pos_ptr, cs_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                            q_ptr, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                            ROPE: tl.constexpr, SCALED: tl.constexpr, NPOS: tl.constexpr, HP: tl.constexpr):
    """Program t: every head of token t as one [HP, D/2] tile (v19); head h < HQ a query head, < HQ + HKV a key head,
    < HQ + 2*HKV a value head (rows above masked). Per-head math as v17's one-program-per-head kernel."""
    t = tl.program_id(0)
    h = tl.arange(0, HP)[:, None]
    i = tl.arange(0, D // 2)[None, :]
    live = h < HQ + 2 * HKV
    isq = h < HQ
    isk = (h >= HQ) & (h < HQ + HKV)
    isv = (h >= HQ + HKV) & live
    base = qkv_ptr + t * stride_qkv + h * D
    e = tl.load(base + 2 * i, mask=live, other=0.0).to(tl.float32)
    o = tl.load(base + 2 * i + 1, mask=live, other=0.0).to(tl.float32)
    r = tl.rsqrt((tl.sum(e * e, 1) + tl.sum(o * o, 1)) / D + eps)[:, None]
    nq = isq | isk
    e = tl.where(nq, e * r, e)
    o = tl.where(nq, o * r, o)
    if ROPE:
        pos = tl.minimum(tl.load(pos_ptr + t).to(tl.int64), NPOS - 1)  # table bound (r6 n4)
        c = tl.load(cs_ptr + pos * D + i)
        sn = tl.load(cs_ptr + pos * D + D // 2 + i)
        e, o = tl.where(nq, e * c - o * sn, e), tl.where(nq, e * sn + o * c, o)
    tl.store(q_ptr + t * (HQ * D) + h * D + 2 * i, e.to(q_ptr.dtype.element_ty), mask=isq)
    tl.store(q_ptr + t * (HQ * D) + h * D + 2 * i + 1, o.to(q_ptr.dtype.element_ty), mask=isq)
    e = tl.where(isk, e.to(tl.bfloat16).to(tl.float32), e)
    o = tl.where(isk, o.to(tl.bfloat16).to(tl.float32), o)
    if SCALED:
        sc = tl.where(isk, tl.load(ks_ptr), tl.load(vs_ptr))
        e = e / sc
        o = o / sc
    row = tl.load(loc_ptr + t).to(tl.int64)
    kvh = tl.where(isk, h - HQ, h - HQ - HKV)
    dst = tl.where(isk, kc_ptr, vc_ptr) + row * (HKV * D) + kvh * D
    kv = isk | isv
    tl.store(dst + 2 * i, e.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=kv)
    tl.store(dst + 2 * i + 1, o.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=kv)


''')
sub('''    _norm_rope_store_kernel[(T, hq + 2 * hkv)](''', '''    _norm_rope_store_kernel[(T,)](''')
sub('''SCALED=cfg["kv_scale"] != 1.0, NPOS=ROPE_POSITIONS, num_warps=1)''',
    '''SCALED=cfg["kv_scale"] != 1.0, NPOS=ROPE_POSITIONS,
        HP=triton.next_power_of_2(hq + 2 * hkv), num_warps=2)''')
sub('''            return None  # FA3 (split-KV head kernel 0.67-0.69x of it at b16/32, check 28ad328cf5284948)
        return ("head", 64, 4, 3, 4, False)''', '''            return None  # FA3 (split-KV head kernel 0.67-0.69x of it at b16/32, check 28ad328cf5284948)
        if kind == "sliding":
            return ("head", 64, 4, 3, 2, False)  # ~2 programs/SM (v19): NSPLIT 4 at b32, 8 at b1/b16
        return ("head", 64, 4, 3, 4, False)''')
sub('''#   sliding verify (q_len 8), any batch      split-KV "head" kernel (FP8 Q.K): 1.15-1.22x FA3''',
    '''#   sliding verify (q_len 8), any batch      split-KV "head" kernel (FP8 Q.K): 1.15-1.22x FA3; ~2 programs/SM (v19)''')
sub('# okc: problem=muse_attention_decode\n"""v17:', '# okc: problem=muse_attention_decode\n'
    '"""v19: v17 + one norm/RoPE/cache-store program per token (all heads as one tile) and ~2 split-KV programs\n'
    'per SM on sliding verify (NSPLIT 4 at batch 32), see tools/build_attn_v19.py.\n\nv17:')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
