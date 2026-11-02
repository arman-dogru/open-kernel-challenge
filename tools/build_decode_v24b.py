"""Build muse_decode_layer/runs/v24b_moe_cfg.py: decode v24 (runs/v24_rows_occ4.py) with the CUDA experts' launch
configs at 32 / 128 / 256 tokens from diagnostic check a6f4e782a8fc4db3's in-prepare sweep (judge weights, cold L2):
T 32 154.6 -> 148.7 us, T 128 286.4 -> 283.4, T 256 408.3 -> 401.9. Same kernels and math (split-K order changes
the FP32 sum order only). Run from the repo root."""
SRC, DST = "muse_decode_layer/runs/v24_rows_occ4.py", "muse_decode_layer/runs/v24b_moe_cfg.py"
s = open(SRC).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


s = rep(s, '''SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2)), 256: (2, (8, 4, 2), (8, 4, 2))}''',
        '''# v24b: 32 / 128 / 256 from decode check a6f4e782a8fc4db3's in-prepare sweep (T 32 154.6 -> 148.7 us, 128 286.4 ->
# 283.4, 256 408.3 -> 401.9; judge weights, cold L2).
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 2, 2), (16, 2, 2)), 128: (2, (8, 4, 3), (16, 4, 2)), 256: (2, (8, 4, 3), (16, 4, 2))}''')
s = rep(s, '# okc: problem=muse_decode_layer\n"""v24:', '''# okc: problem=muse_decode_layer
"""v24b: v24 with the CUDA experts' launch configs at 32 / 128 / 256 tokens re-picked from an in-prepare sweep on the
judge (check a6f4e782a8fc4db3): 32 tokens split 2 (was 4) and 16 warps for GEMM 2; 128 / 256 prefetch 3 for GEMM 1
and 16 warps for GEMM 2 (tools/build_decode_v24b.py).

v24:''')
open(DST, "w").write(s)
print("wrote", DST)
