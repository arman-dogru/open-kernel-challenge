"""Build muse_decode_layer/runs/v25_ws4_fv.py: decode v24b (runs/v24b_moe_cfg.py, check 55385a5e40244c4e, 1.392x) with
full verify (q_len 8, batch >= 2) on attention v25e's warp-specialized kernel attn_fvws (runs/v25e_ws4.py, attention
check 71c0267525f7412e: b16 / b32 L8 FV 170.9 / 325.7 -> 143.2 / 268.6 us).

The decode file's _FV_SRC is byte-identical to attention v24's, so the whole FV block (sources, flags, _fv_smem /
_wsf / _ws_opts, _fv_cubin / _fv_fn) is taken from v25e_ws4.py verbatim (it carries the "memory" clobbers on the
inline ld.shared / st.shared asm), and decode's _fv launch gets v25e's block size and shared-memory size. FV_OCC 3
(splits per SM) comes with the block, as in v25e. Run from the repo root."""
SRC = "muse_decode_layer/runs/v24b_moe_cfg.py"
ATT = "muse_attention_decode/runs/v25e_ws4.py"
DST = "muse_decode_layer/runs/v25_ws4_fv.py"
s = open(SRC).read()
a = open(ATT).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


def block(text):
    i = text.index('_FV_SRC = r"""')
    j = text.index("\n\n\ndef _fv_sm90(", i)
    return text[i:j]


old, new = block(s), block(a)
assert old == block(open("muse_attention_decode/runs/v24_rows_occ4.py").read()), "decode FV block is not attention v24's"
new = rep(new, 'ROWS_FULL = ("rows", 128, 4, 3, 4, True)  # full decode, batch >= 2\n', "")
s = rep(s, old, new)
s = rep(s, "        f((nsplit, batch), (256,), (", "        f((nsplit, batch), (256 + 32 * _ws_npw(FV_VT1) if FV_VT1 & 64 else 256,), (")
s = rep(s, '    """Full verify at batch >= 2: CUDA split-KV partials (tools/attn_v23_fv.cu), then the Triton combine."""',
        '    """Full verify at batch >= 2: CUDA split-KV partials (v25: attn_fvws, tools/attn_v25_ws.cu), then the Triton\n'
        '    combine."""')
s = rep(s, '# okc: problem=muse_decode_layer\n"""v24b:', '''# okc: problem=muse_decode_layer
"""v25: v24b with full verify (q_len 8, batch >= 2) on attention v25e's warp-specialized kernel attn_fvws: 4 cp.async
producer warps (setmaxnreg 24) + 2 consumer warpgroups (setmaxnreg 104), 80-register launch so 2 CTAs fit per SM,
per-warpgroup barriers, ST 3, 3 splits per SM (flags 55869504). Attention check 71c0267525f7412e: FV b16 / b32 170.9 /
325.7 -> 143.2 / 268.6 us; bit-exact vs v24's FV at equal splits (diag 5ff307988ce14518, 648/648). The FV block is
attention v25e_ws4.py's verbatim ("memory" clobbers on the inline shared-memory asm). tools/build_decode_v25.py.

v24b:''')
open(DST, "w").write(s)
print("wrote", DST)
