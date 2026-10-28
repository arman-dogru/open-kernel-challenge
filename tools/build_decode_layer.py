"""Builds muse_decode_layer/submission.py from SGLang's baseline.py plus our kernels from the
other problems (single-file submission), swapping in each piece only where its own problem's
H200 runs showed a win. Run from the repo root."""
import re

A = open("muse_attention_decode/submission.py").read()
B = open("muse_bf16_proj/submission.py").read()
base = open("muse_decode_layer/baseline.py").read()


def grab(src, start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


attn_kernels = grab(A, "LOG2E = tl.constexpr", "def _qk_norm(")
bf16_kernel = grab(B, "@triton.jit\ndef _swapab_gemm", "# (BN, BK, SPLIT_K")

header = '''# okc: problem=muse_decode_layer
"""Muse decoder layer: SGLang's layer (baseline.py, kept below) with our kernels swapped in where
their own problems' H200 runs showed a win:

  attention       split-KV Triton decode kernel + combine fused with the per-head norm x gate
                  (muse_attention_decode), on the shapes in SPLITKV_ON; FA3 elsewhere
  o, pooler       swap-AB Triton GEMM up to 32 tokens (muse_bf16_proj); cuBLAS above
  router          BF16 hi + lo Triton split-K GEMM for 9-32 tokens (SGLang runs the same hi/lo
                  split above 8 tokens, as two cuBLAS GEMMs); FP32 F.linear up to 8 as SGLang

Everything else (norms, routing, Marlin experts, unpooler) is SGLang's, unchanged.

--- SGLang baseline notes ---
'''
body = base.replace('"""SGLang\'s decoder layer', header + 'SGLang\'s decoder layer', 1)
body = body.replace("from typing import TypedDict\n", "from typing import TypedDict\n\nimport triton\nimport triton.language as tl\n", 1)

ours = '''

# ---- our kernels -----------------------------------------------------------------------------
''' + attn_kernels + bf16_kernel + '''
# (BN, BK, SPLIT_K, warps, stages): muse_bf16_proj v3 configs
PROJ_CFG = {"o": (64, 128, 1, 4, 4), "pooler": (32, 256, 1, 4, 4), "router": (16, 256, 8, 4, 4)}
PROJ_MAX_TOKENS = 32
# (layer kind, q_len, batch) where the split-KV kernel beat FA3 on muse_attention_decode
SPLITKV_ON = SPLITKV_ON_PLACEHOLDER


def _proj(x, w, layer, lo=None):
    x = x.contiguous()
    m, k = x.shape
    n = w.shape[0]
    bn, bk, split, warps, stages = PROJ_CFG[layer]
    bm = 16 if m <= 16 else 32
    if split > 1:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _swapab_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm), split)](
        x, w, w if lo is None else lo, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_LO=lo is not None, OUT_BF16=split == 1,
        num_warps=warps, num_stages=stages)
    return out


def _nsplit(cfg, batch, hkv, capacity):
    span = (cfg["window"] + cfg["q_len"] - 1) if cfg["window"] else capacity
    want = max(1, (132 * 4) // (batch * hkv))
    n = min(want, max(1, span // 256), 256)
    return 1 << (n.bit_length() - 1)


def _splitkv_attention(data, q, qkv, cfg, gate):
    """q [T, Hq*64] after norm/RoPE (cache already written) -> rmsnorm(attn) * sigmoid(gate), [T, Hq*64]."""
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = batch * L
    G = hq // hkv
    page = cfg["page_size"]
    capacity = data["page_table"].shape[1] * page
    nsplit = _nsplit(cfg, batch, hkv, capacity)
    span = (cfg["window"] + L - 1) if cfg["window"] else capacity
    chunk = triton.cdiv(span, nsplit)
    o_part = torch.empty((T * hq * nsplit, d), device=q.device, dtype=torch.float32)
    lse = torch.empty((T * hq * nsplit,), device=q.device, dtype=torch.float32)
    args = (q, q.stride(0), data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8),
            data["page_table"], data["page_table"].stride(0), data["cache_seqlens"], data["k_scale"],
            data["v_scale"], o_part, lse, cfg["sm_scale"] * cfg["qk_scale_factor"], chunk, page)
    if hkv * d <= 128 and not cfg["window"]:
        rows = max(16, triton.next_power_of_2(hq * L))
        _attn_split_rows[(nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"], BLOCK_M=rows, BLOCK_N=64, D=d,
            HW=True, num_warps=4 if rows <= 16 else 8, num_stages=3)
    else:
        _attn_split[(hkv, nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"],
            BLOCK_M=max(16, triton.next_power_of_2(G * L)), BLOCK_N=64, D=d, HW=True, num_warps=4, num_stages=3)
    out = torch.empty((T, hq * d), device=q.device, dtype=torch.bfloat16)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, gate.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=1)
    return out
'''
body = body.replace("\n\ndef _gemm(x, w):", ours + "\n\ndef _gemm(x, w):", 1)

# prepare: non-transposed hi/lo for the Triton router
body = body.replace('''        "router_lo_t": lo.t().contiguous(),
''', '''        "router_lo_t": lo.t().contiguous(),
        "router_hi": hi.contiguous(),
        "router_lo": lo.contiguous(),
''', 1)

# attention swap
old_attn = body[body.index("    window = cfg[\"window\"]\n    descale = {}"):body.index("    attn_output = _gemm(o, dense[\"o\"])")]
new_attn = '''    window = cfg["window"]
    kind = "sliding" if window else "full"
    if (kind, length, batch) in SPLITKV_ON and hkv * d <= 128:
        o = _splitkv_attention(data, q.reshape(tokens, hq * d), qkv, cfg, gate)
    else:
''' + "\n".join(("    " + l) if l.strip() else l for l in old_attn.split("\n")[1:]) 
body = body.replace(old_attn, new_attn.rstrip(" ") , 1)
body = body.replace('''    attn_output = _gemm(o, dense["o"])  # all-reduce: identity''', '''    if tokens <= PROJ_MAX_TOKENS:
        attn_output = _proj(o, dense["o"], "o")  # all-reduce: identity
    else:
        attn_output = _gemm(o, dense["o"])  # all-reduce: identity''', 1)
body = body.replace('''    expert_input = _gemm(ffn_input, dense["pooler"])''', '''    if tokens <= PROJ_MAX_TOKENS:
        expert_input = _proj(ffn_input, dense["pooler"], "pooler")
    else:
        expert_input = _gemm(ffn_input, dense["pooler"])''', 1)
body = body.replace('''        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)''', '''        if split_router and tokens <= PROJ_MAX_TOKENS:
            logits = _proj(ffn_input, state["router_hi"], "router", lo=state["router_lo"])
        elif split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)''', 1)
import sys
on = sys.argv[1] if len(sys.argv) > 1 else "set()"
body = body.replace("SPLITKV_ON_PLACEHOLDER", on)
open("muse_decode_layer/submission.py", "w").write(body)
print("wrote muse_decode_layer/submission.py")
