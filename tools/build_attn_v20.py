"""Build muse_attention_decode/runs/v20_verify_tiles.py from v17_rows_prefetch.py + v19_tok_norm.py (byte-exact).

v20 = v17 + (a) from 64 tokens (verify 16/32) v19's one-program-per-token norm/RoPE/cache-store kernel (v19b check
3d618fc619024323: verify 16 2.8 -> 2.1, verify 32 4.4 -> 2.9 us; v17's merge kept) + (b) sliding verify on the head
kernel with 128-key tiles, the V gather issued with K (HOIST) and the next tile's page indices loaded one iteration
ahead (PREF, as v17's rows kernel). A sliding verify block needs 2,048 + 7 keys, so 8 power-of-two splits always
leave a remainder (v17: 257 keys = 4 full 64-key tiles + a 1-key tile); the kernel is latency-bound (one wave, each
tile a dependent page-index -> K -> V chain), so (b) cuts the serial steps per split from 5 to 3 and overlaps them.
"""
import os, sys
SRC, V19 = "muse_attention_decode/runs/v17_rows_prefetch.py", "muse_attention_decode/runs/v19_tok_norm.py"
DST = "muse_attention_decode/runs/v20_verify_tiles.py"
VERIFY_CFG = os.environ.get("V20_VERIFY_CFG", '("head", 128, 4, 3, 4, True)')  # offline register probes only
s = open(SRC).read()
v19 = open(V19).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# (a) per-token norm kernel from 64 tokens (as v19b, merge unchanged)
a = v19.index("def _norm_rope_store_kernel(")
tok = v19[a:v19.index("def _norm_rope_store_fused(")].replace("def _norm_rope_store_kernel(", "def _norm_rope_store_tok_kernel(")
tok = tok.replace("(v19); head h", "(v19, used at T >= TOK_MIN); head h")
sub("def _norm_rope_store_fused(", "@triton.jit\n" + tok +
    "TOK_MIN = 64  # tokens from which the per-token norm/RoPE/store kernel is used (v19b / v20)\n\n\n"
    "def _norm_rope_store_fused(")
sub('''    _norm_rope_store_kernel[(T, hq + 2 * hkv)](
        qkv, qkv.stride(0), data["positions"], cs, data["cache_loc"],
        data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
        q, RMS_EPS, HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=cfg["kv_scale"] != 1.0, NPOS=ROPE_POSITIONS, num_warps=1)''',
    '''    args = (qkv, qkv.stride(0), data["positions"], cs, data["cache_loc"],
            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
            q, RMS_EPS)
    if T >= TOK_MIN:
        _norm_rope_store_tok_kernel[(T,)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=cfg["kv_scale"] != 1.0,
                                          NPOS=ROPE_POSITIONS, HP=triton.next_power_of_2(hq + 2 * hkv), num_warps=2)
    else:
        _norm_rope_store_kernel[(T, hq + 2 * hkv)](*args, HQ=hq, HKV=hkv, D=d, ROPE=rope,
                                                   SCALED=cfg["kv_scale"] != 1.0, NPOS=ROPE_POSITIONS, num_warps=1)''')

# (b) head kernel: PREF (next tile's page indices one iteration ahead)
sub('''                D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False):
    h = tl.program_id(0)''', '''                D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False,
                PREF: tl.constexpr = False):
    h = tl.program_id(0)''')
sub('''    acc = tl.zeros((BLOCK_M, D), tl.float32)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
        row = (page.to(tl.int64) * PAGE + t % PAGE) * (HKV * D) + h * D''',
    '''    acc = tl.zeros((BLOCK_M, D), tl.float32)
    if PREF:  # the first tile's page indices; each iteration then loads the next tile's (v20, as _attn_split_rows)
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
        row = (page.to(tl.int64) * PAGE + t % PAGE) * (HKV * D) + h * D''')
sub('''            HW=hw, HOIST=hoist, PV16=hw, num_warps=warps, num_stages=stages)
    gate''', '''            HW=hw, HOIST=hoist, PREF=hoist, PV16=hw, num_warps=warps, num_stages=stages)
    gate''')
sub('''            return None  # FA3 (split-KV head kernel 0.67-0.69x of it at b16/32, check 28ad328cf5284948)
        return ("head", 64, 4, 3, 4, False)''', '''            return None  # FA3 (split-KV head kernel 0.67-0.69x of it at b16/32, check 28ad328cf5284948)
        if kind == "sliding":  # v20: 128-key tiles (3 serial steps per 257-key split, not 5), V hoisted, PREF
            return ''' + VERIFY_CFG + '''
        return ("head", 64, 4, 3, 4, False)''')
sub('# okc: problem=muse_attention_decode\n"""v17:', '# okc: problem=muse_attention_decode\n'
    '"""v20: v17 + the per-token norm/RoPE/store kernel from 64 tokens and sliding verify on 128-key head-kernel\n'
    'tiles with V hoisted and next-tile page-index prefetch. See tools/build_attn_v20.py.\n\nv17:')
if '"head", 64,' in VERIFY_CFG:  # v20b (tools/build_attn_v20b.py): describe the 64-key tiles it builds
    s = s.replace("sliding verify on 128-key head-kernel\ntiles with V hoisted", "sliding verify on 64-key head-kernel\ntiles (v20b) with V hoisted")
    s = s.replace("# v20: 128-key tiles (3 serial steps per 257-key split, not 5), V hoisted, PREF",
                  "# v20b: v17's 64-key tiles, V hoisted, PREF (128-key tiles were slower)")
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
