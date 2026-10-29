import re
S = open("muse_decode_layer/runs/v12_precompile.py").read()
A = open("muse_attention_decode/runs/v14b_docfix.py").read()
def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:60], s.count(old))
    return s.replace(old, new)
def grab(src, a, b):
    i = src.index(a); j = src.index(b, i); return src[i:j]

# 1. split kernels with PV16 (v14), replacing v12's copies
old_k = grab(S, "@triton.jit\ndef _attn_split(", "@triton.jit\ndef _combine_norm_gate")
new_k = grab(A, "@triton.jit\ndef _attn_split(", "@triton.jit\ndef _combine_norm_gate")
S = rep(S, old_k, new_k)

# 2. fused norm+RoPE+FP8+cache-write kernel (v14), with the RoPE table read guarded (r6 n4)
fk = grab(A, "@triton.jit\ndef _norm_rope_store_kernel", "def _norm_rope_store_fused")
fk = rep(fk, "q_ptr, eps, HQ: tl.constexpr,", "q_ptr, eps, NPOS, HQ: tl.constexpr,")
fk = rep(fk, "            pos = tl.load(pos_ptr + t).to(tl.int64)\n",
         "            # r6 n4: clamp to the table (2^21 rows, as SGLang's); judge positions stay < 2^20 + 9\n"
         "            pos = tl.minimum(tl.load(pos_ptr + t).to(tl.int64), NPOS - 1)\n")
fused_py = '''

def _norm_rope_store_fused(data, qkv, cfg, cos_sin):
    """attention v14's fused step: per-head RMSNorm, interleaved RoPE, q out in FP8 E4M3 (one RNE,
    saturating fp32 -> E4M3 rounding, as FA3's own cast), k (BF16-rounded, / k_scale) and v (/ v_scale)
    cast to E4M3 and written to their cache rows. Replaces FusedQKNorm + the casts + store_cache."""
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = qkv.shape[0]
    q = torch.empty((T, hq * d), device=qkv.device, dtype=FP8)
    rope = bool(cfg["rope_theta"])
    _norm_rope_store_kernel[(T, hq + 2 * hkv)](
        qkv, qkv.stride(0), data["positions"], cos_sin, data["cache_loc"],
        data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
        q, RMS_EPS, cos_sin.shape[0], HQ=hq, HKV=hkv, D=d, ROPE=rope, SCALED=cfg["kv_scale"] != 1.0, num_warps=1)
    return q


'''
S = rep(S, "@triton.jit\ndef _combine_norm_gate", fk.rstrip() + "\n" + fused_py + "@triton.jit\ndef _combine_norm_gate")

# 3. split-KV launches: P.V in BF16 with FP8 q
S = rep(S, "HW=True, HOIST=hoist,", "HW=True, HOIST=hoist, PV16=True,", 2)
S = rep(S, '"""q [T, Hq*64] after norm/RoPE (cache already written)', '"""q [T, Hq*64] FP8 E4M3 after norm/RoPE (cache already written)')

# 4. kernel_fn attention prologue
old = grab(S, "    if cfg[\"rope_theta\"]:\n        q, k = state[\"qk_norm\"]", "    window = cfg[\"window\"]\n")
S = rep(S, old, '''    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    # One fused launch: norms, RoPE, FP8 q, K/V cache rows (slot 0 is a live row, reserved_skip_index=-1).
    q = _norm_rope_store_fused(data, qkv, cfg, state["cos_sin"])
''')
S = rep(S, "q=q.view(tokens, hq, d).to(FP8),", "q=q.view(tokens, hq, d),  # already E4M3")
S = rep(S, '        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),\n', "")
S = rep(S, "from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm\n", "")
S = rep(S, "from sglang.kernels.ops.kvcache.kvcache import store_cache\n", "")

# 5. dead code: SMALL_MAX, SMALL_CFG[32]
S = rep(S, grab(S, "SMALL_MAX = 32", "# (NB, (WARPS"), "")
S = rep(S, "SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),\n             32: (1, (8, 4, 2), (8, 4, 2))}",
        "# The layer uses these CUDA experts up to LAYER_SMALL_MAX = 16 tokens only, so v24's 32-token entry is not carried.\n"
        "SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2))}")
# r4 nit 6: alignment assert; nit 9: FP32 intermediates
S = rep(S, "    bs, H = x.shape\n    I = cfg[\"d_expert\"]",
        "    assert x.data_ptr() % 16 == 0, \"moe_mma reads hidden_states as uint4\"  # r4 n6 (metadata: graph-safe)\n"
        "    bs, H = x.shape\n    I = cfg[\"d_expert\"]")
S = rep(S, "    h32 = buf[:NR * 2 * I]\n",
        "    # h32 / y32: FP32 intermediates (higher precision than SGLang's BF16 caches) (r4 n9)\n    h32 = buf[:NR * 2 * I]\n")

# 6. docstring
S = rep(S, '# okc: problem=muse_decode_layer\n"""v12:', '''# okc: problem=muse_decode_layer
"""v13: v12 plus muse_attention_decode v14's attention front end (r6-approved, submitted as v14b):
  * one fused Triton norm + interleaved RoPE + FP8 + cache-write program per (token, head) replaces
    FusedQKNorm, the K/V/q casts and store_cache (4-5 launches) on every path; the RoPE table read is
    clamped to the table (r6 n4);
  * the query comes out of it in FP8 E4M3 (one RNE, saturating fp32 -> E4M3 rounding, as FA3 itself
    casts), so the FA3 verify path takes it as is (no separate cast launch), and the split-KV decode
    kernels run Q.K^T as an FP8 x FP8 tensor-core dot on the raw cache bytes; P.V stays BF16, softmax
    and accumulation FP32.
Not carried from v14: its dead SGLang wrappers (_qk_norm, _norm_rope_store, _norm_rope_store_torch).
Cleanups: SMALL_MAX and SMALL_CFG[32] removed (dead: the layer's CUDA experts stop at 16 tokens);
r4 n6 (hidden_states alignment assert), r4 n9 (FP32 intermediates note).

v12:''')
S = S.replace("  FusedQKNorm (Triton)             per-head RMSNorm", "  FusedQKNorm (Triton)             per-head RMSNorm", 1)
open("muse_decode_layer/runs/v13_fused_attn.py", "w").write(S)
