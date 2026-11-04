"""Build muse_decode_layer/runs/v23_fv_attn.py from decode runs/v19_nb2_256_evict.py (board 1.364x) + attention
v23's full-verify CUDA path (muse_attention_decode/runs/v23_fv_addr.py, board 1.478x). Run from the repo root.

Full-attention verify blocks (q_len 8) at batch >= 2 leave FA3 + FusedRMSNormGate for attention v23's sm_90a
split-KV kernel (tools/attn_v23_fv.cu: FP8 wgmma Q.K^T, cp.async page-table gather, address math hoisted, Q.K^T
before the V transpose) and the Triton combine that applies the per-head RMSNorm and sigmoid gate. The kernel
source, NVRTC compile (at import, never inside graph capture), CuPy load, device guard and launch are copied
byte-for-byte from attention v23; everything else is v19 unchanged. Other GPUs / head layouts / non-power-of-two
pages keep FA3.
"""
import sys

DEC = "muse_decode_layer/runs/v19_nb2_256_evict.py"
ATT = "muse_attention_decode/runs/v23_fv_addr.py"
DST = "muse_decode_layer/runs/v23_fv_attn.py"
s = open(DEC).read()
a = open(ATT).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


# 1) attention v23's CUDA block: _FV_SRC .. _fv_page_ok (verbatim), plus _fv with decode's gate tensor.
i = a.index('_FV_SRC = r"""')
j = a.index("def _fv(data, q, qkv, cfg):")
block = a[i:j]
k = a.index("def kernel_fn(data):")
fv = a[j:k]
fv = rep(fv, "def _fv(data, q, qkv, cfg):", "def _fv(data, q, gate, cfg):")
fv = rep(fv, '''    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = torch.empty((T, hq * d), device=qkv.device, dtype=data["qkv"].dtype)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=4 if nsplit >= 64 else 1)
    return out.view(batch, L, hq * d)''', '''    out = torch.empty((T, hq * d), device=q.device, dtype=torch.bfloat16)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, gate.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=4 if nsplit >= 64 else 1)
    return out''')
fv = fv.replace("(tools/attn_v22_fv.cu)", "(tools/attn_v23_fv.cu)")
s = rep(s, "\ndef kernel_fn(data, state):", "\n# Full verify at batch >= 2: attention v23's CUDA path, verbatim.\n" + block + fv + "\ndef kernel_fn(data, state):")

# 2) dispatch: before FA3.
s = rep(s, '''    if path is not None:
        o = _splitkv_attention(data, q.reshape(tokens, hq * d), qkv, cfg, gate, *path)
    else:''', '''    if path is None and _fv_ok(kind, length, batch, hq, hkv, d, q) and _fv_page_ok(cfg["page_size"]):
        o = _fv(data, q.reshape(tokens, hq * d), gate, cfg)  # v23: CUDA full-verify kernel instead of FA3
    elif path is not None:
        o = _splitkv_attention(data, q.reshape(tokens, hq * d), qkv, cfg, gate, *path)
    else:''')

# 3) header.
s = rep(s, '# okc: problem=muse_decode_layer\n"""v19:', '''# okc: problem=muse_decode_layer
"""v23: v19 with full-attention verify blocks at batch >= 2 on attention v23's CUDA sm_90a split-KV kernel (FP8
wgmma Q.K^T, cp.async page-table gather; tools/attn_v23_fv.cu, attention board 1.478x) plus the Triton combine
with the per-head RMSNorm and sigmoid gate, instead of FA3 + FusedRMSNormGate. Kernel, NVRTC compile at import,
CuPy load and launch copied from muse_attention_decode/runs/v23_fv_addr.py (tools/build_decode_v23.py). In decode
v19's profile FA3 took 188.5 / 374.1 us at verify 16 / 32 full; attention v23's kernel ran those shapes in ~162 /
~324 us on the attention problem.

v19:''')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
