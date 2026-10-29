"""Build muse_attention_decode/runs/v19b_tok_launch.py from v17_rows_prefetch.py and v19_tok_norm.py (byte-exact).

v19 check 3b5207bbc0d042c8 (1.366x) per kernel vs v17 8deec285efbe4b57: the one-program-per-token norm/RoPE/store
kernel wins at T = 128 / 256 tokens (verify 16 / 32: 2.8 -> 2.3, 4.4 -> 2.8 us) and loses 0.4-0.7 us at T <= 32
(decode); ~2 programs/SM on sliding verify b32 did not help (split 16.2 -> 16.6, combine 4.0 -> 3.8 us), dropped.
v19b = v17 + by token count (regime, not exact batch): T >= 64 -> v19's per-token norm/RoPE/store kernel, and the
split merge as 4 heads per program ([4, NSPLIT, D] tile, grid (T, HQ / 4)) instead of one (token, head) per
program (4,096 one-warp programs at T = 256). T < 64: v17's kernels unchanged. Same math per head.
"""
import sys
SRC, V19, DST = ("muse_attention_decode/runs/v17_rows_prefetch.py", "muse_attention_decode/runs/v19_tok_norm.py",
                 "muse_attention_decode/runs/v19b_tok_launch.py")
s = open(SRC).read()
v19 = open(V19).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


a = v19.index("def _norm_rope_store_kernel(")
tok = v19[a:v19.index("def _norm_rope_store_fused(")].replace("def _norm_rope_store_kernel(", "def _norm_rope_store_tok_kernel(")
tok = tok.replace("(v19); head h", "(v19, used at T >= 64); head h")
TOK_MIN = '''TOK_MIN = 64  # tokens from which the per-token norm kernel and the 4-heads-per-program merge are used (v19b)


'''
sub("def _norm_rope_store_fused(", "@triton.jit\n" + tok + TOK_MIN + "def _norm_rope_store_fused(")
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
sub('''def _cos_sin(theta, d, device):''', '''@triton.jit
def _combine_norm_gate_hb(o_ptr, lse_ptr, gate_ptr, stride_gt, out_ptr, eps,
                          HQ: tl.constexpr, NSPLIT: tl.constexpr, D: tl.constexpr, HB: tl.constexpr):
    """_combine_norm_gate for HB heads of one token per program (v19b, T >= TOK_MIN): [HB, NSPLIT, D] tile."""
    t = tl.program_id(0)
    h0 = tl.program_id(1) * HB
    hb = tl.arange(0, HB)
    sp = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    idx = (t * HQ + h0 + hb[:, None]) * NSPLIT + sp[None, :]  # [HB, NSPLIT]
    lse = tl.load(lse_ptr + idx)
    mx = tl.max(lse, 1)
    w = tl.where(lse == float("-inf"), 0.0, tl.exp2(lse - mx[:, None]))
    o = tl.load(o_ptr + idx[:, :, None] * D + d[None, None, :])
    a = tl.sum(o * w[:, :, None], 1) / tl.sum(w, 1)[:, None]  # [HB, D]
    r = a * tl.rsqrt(tl.sum(a * a, 1) / D + eps)[:, None]
    col = (h0 + hb[:, None]) * D + d[None, :]
    g = tl.load(gate_ptr + t * stride_gt + col).to(tl.float32)
    tl.store(out_ptr + t * (HQ * D) + col, (r * tl.sigmoid(g)).to(out_ptr.dtype.element_ty))


def _cos_sin(theta, d, device):''')
sub('''    _combine_norm_gate[(T, hq)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=4 if nsplit >= 64 else 1)''',
    '''    if T >= TOK_MIN and hq % 4 == 0 and nsplit <= 32:
        _combine_norm_gate_hb[(T, hq // 4)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit,
                                            D=d, HB=4, num_warps=2)
    else:
        _combine_norm_gate[(T, hq)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                    num_warps=4 if nsplit >= 64 else 1)''')
sub('# okc: problem=muse_attention_decode\n"""v17:', '# okc: problem=muse_attention_decode\n'
    '"""v19b: v17 + from 64 tokens (verify 16/32) one norm/RoPE/cache-store program per token and the split merge\n'
    'as 4 heads per program; below 64 tokens v17 unchanged. See tools/build_attn_v19b.py.\n\nv17:')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
