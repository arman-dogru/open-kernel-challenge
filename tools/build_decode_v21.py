"""Build muse_decode_layer/runs/v21_attn_v20b.py from v20_rows_prefetch.py (byte-exact, rerunnable).

decode v21 = decode v20 + attention v20b's two wins (check 278e0857f453409b, 1.406x; sliding verify v1/v16/v32
1.573/1.536/1.278x -> 1.750/1.598/1.374x vs attention v17): (a) from 64 tokens (verify 16/32) one norm/RoPE/cache-store
program per token over all heads (attention v19's kernel, here with decode's runtime NPOS argument), and (b) sliding
verify on the split-KV head kernel with the V gather issued with K (HOIST) and the next tile's page indices loaded one
iteration ahead (PREF, as the rows kernel). Same math per head; the split merge is unchanged.
"""
import sys
SRC, DST = "muse_decode_layer/runs/v20_rows_prefetch.py", "muse_decode_layer/runs/v21_attn_v20b.py"
s = open(SRC).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


TOK = '''@triton.jit
def _norm_rope_store_tok_kernel(qkv_ptr, stride_qkv, pos_ptr, cs_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                                q_ptr, eps, NPOS, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                                ROPE: tl.constexpr, SCALED: tl.constexpr, HP: tl.constexpr):
    """Program t: every head of token t as one [HP, D/2] tile (attention v19, used from TOK_MIN tokens); head h < HQ a
    query head, < HQ + HKV a key head, < HQ + 2*HKV a value head (rows above masked). Math as _norm_rope_store_kernel."""
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


TOK_MIN = 64  # tokens from which the per-token norm/RoPE/store kernel is used (attention v19b / v20b, decode v21)


def _norm_rope_store_fused('''
sub("def _norm_rope_store_fused(", TOK)
sub('''    _norm_rope_store_kernel[(T, hq + 2 * hkv)](
        qkv, qkv.stride(0), data["positions"], cos_sin, data["cache_loc"],
        data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
        q, RMS_EPS, cos_sin.shape[0], HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=cfg["kv_scale"] != 1.0, num_warps=1)''',
    '''    args = (qkv, qkv.stride(0), data["positions"], cos_sin, data["cache_loc"],
            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
            q, RMS_EPS, cos_sin.shape[0])
    if T >= TOK_MIN:
        _norm_rope_store_tok_kernel[(T,)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=cfg["kv_scale"] != 1.0,
                                          HP=triton.next_power_of_2(hq + 2 * hkv), num_warps=2)
    else:
        _norm_rope_store_kernel[(T, hq + 2 * hkv)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope,
                                                   SCALED=cfg["kv_scale"] != 1.0, num_warps=1)''')
# head kernel: PREF
a = s.index("def _attn_split(")
head_sig_end = s.index("):\n", a)
sig = s[a:head_sig_end]
assert sig.endswith("PV16: tl.constexpr = False"), sig[-60:]
s = s[:head_sig_end] + ",\n                PREF: tl.constexpr = False" + s[head_sig_end:]
b = s.index("def _attn_split_rows(")
head = s[a:b]
old_loop = '''    acc = tl.zeros((BLOCK_M, D), tl.float32)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
'''
assert head.count(old_loop) == 1, head[:200]
new_loop = '''    acc = tl.zeros((BLOCK_M, D), tl.float32)
    if PREF:  # the first tile's page indices; each iteration then loads the next tile's (decode v21, as the rows kernel)
        t0 = lo + tl.arange(0, BLOCK_N)
        pnext = tl.load(pt_ptr + b * stride_pt + t0 // PAGE, mask=t0 < hi, other=0)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        if PREF:
            page = pnext
            tn = t + BLOCK_N
            pnext = tl.load(pt_ptr + b * stride_pt + tn // PAGE, mask=tn < hi, other=0)
        else:
            page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
'''
s = s[:a] + head.replace(old_loop, new_loop) + s[b:]
sub('''            BLOCK_M=max(16, triton.next_power_of_2(G * L)), BLOCK_N=block_n, D=d, HW=True, HOIST=hoist, PV16=True,
            num_warps=warps, num_stages=stages)''', '''            BLOCK_M=max(16, triton.next_power_of_2(G * L)), BLOCK_N=block_n, D=d, HW=True, HOIST=hoist, PREF=hoist,
            PV16=True, num_warps=warps, num_stages=stages)''')
sub('''        if hkv * d > 128 or (kind == "full" and batch >= 2):
            return None
        return ("head", 64, 4, 3, 4, False)''', '''        if hkv * d > 128 or (kind == "full" and batch >= 2):
            return None
        if kind == "sliding":  # decode v21 (attention v20b): V hoisted + next-tile page-index prefetch
            return ("head", 64, 4, 3, 4, True)
        return ("head", 64, 4, 3, 4, False)''')
sub('# okc: problem=muse_decode_layer\n"""v20:', '# okc: problem=muse_decode_layer\n'
    '"""v21: v20 + attention v20b (per-token norm/RoPE/store from 64 tokens; sliding verify head kernel with V hoisted\n'
    'and next-tile page-index prefetch). See tools/build_decode_v21.py.\n\nv20:')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
