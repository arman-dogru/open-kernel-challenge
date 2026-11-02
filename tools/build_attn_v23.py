"""Build muse_attention_decode/runs/v23_fv_addr.py from runs/v22_cuda_fv.py + tools/attn_v23_fv.cu (byte-exact).

v23 = v22 with the full-verify CUDA kernel source replaced by tools/attn_v23_fv.cu: identical math, data layouts,
launch and host code; the kernel's per-thread loader / V-transpose shared-memory offsets are computed once and
pinned in registers, the stage index is a counter, and Q.K^T is issued before the V transpose (see the .cu header).
"""
import sys
SRC, CU, DST = ("muse_attention_decode/runs/v22_cuda_fv.py", "tools/attn_v23_fv.cu",
                "muse_attention_decode/runs/v23_fv_addr.py")
s = open(SRC).read()
cu = open(CU).read()
assert '"""' not in cu and not cu.endswith("\\")
a = s.index('_FV_SRC = r"""\n') + len('_FV_SRC = r"""\n')
b = s.index('"""', a)
s = s[:a] + cu + s[b:]
old = '# okc: problem=muse_attention_decode\n"""v22:'
assert s.count(old) == 1
s = s.replace(old, '# okc: problem=muse_attention_decode\n"""v23: v22 with the full-verify kernel\'s address math hoisted out of '
              'its tile loop and Q.K^T issued\nbefore the V transpose (tools/attn_v23_fv.cu, tools/build_attn_v23.py).\n\nv22:')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
