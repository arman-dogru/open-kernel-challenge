"""Build muse_attention_decode/runs/v21_verify_w8.py (default) or v21b_verify_w8_occ2.py ("b") from build_attn_v20.py.

v20b check 278e0857f453409b (1.406x): sliding verify b32 = split 17.2 + merge 3.9 + norm 2.8 us. The split kernel is
compute/issue-bound there (2.15 GFLOP in 17.2 us ~ 125 TF/s, near this kernel's ~145 ceiling), with 2 programs x 4
warps = 8 warps per SM (214 regs). v21: sliding verify on 8-warp programs capped at 128 registers (maxnreg; offline
sm_90 compile with the judge's dtypes: 127 regs, 0 stack; 4 warps under the same cap spills 424 B), so 2 programs x 8
warps = 16 warps per SM on the same 64-key tiles, V hoisted, next-tile page-index prefetch. The cap is applied only
when this Triton's CUDA options have maxnreg. v21b: also ~2 programs per SM (one wave at b32: NSPLIT 4).
"""
import os, subprocess, sys, tempfile
var = sys.argv[1] if len(sys.argv) > 1 else ""
occ = 2 if var == "b" else 4
DST = sys.argv[2] if len(sys.argv) > 2 else (
    "muse_attention_decode/runs/v21b_verify_w8_occ2.py" if var == "b" else "muse_attention_decode/runs/v21_verify_w8.py")
with tempfile.TemporaryDirectory() as td:
    tmp = os.path.join(td, "v20.py")
    env = dict(os.environ, V20_VERIFY_CFG=f'("head", 64, 8, 3, {occ}, True)')
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "build_attn_v20.py"), tmp], check=True,
                   env=env, stdout=subprocess.DEVNULL)
    s = open(tmp).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


sub("        if kind == \"sliding\":  # v20b: v17's 64-key tiles, V hoisted, PREF (128-key tiles were slower)\n",
    "        if kind == \"sliding\":  # v21: 64-key tiles, 8 warps capped at VERIFY_MAXNREG, V hoisted, PREF"
    + (", ~2 programs/SM" if occ == 2 else "") + "\n")
sub('''def _attn_path(kind, L, batch, hkv, d):''', '''VERIFY_MAXNREG = 128  # sliding verify programs: 8 warps x 128 regs, 2 per SM (v21; 0 stack offline at sm_90)


def _maxnreg_ok():
    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        return "maxnreg" in getattr(CUDAOptions, "__dataclass_fields__", {})
    except Exception:
        return False


_MAXNREG_OK = _maxnreg_ok()


def _attn_path(kind, L, batch, hkv, d):''')
sub('''            HW=hw, HOIST=hoist, PREF=hoist, PV16=hw, num_warps=warps, num_stages=stages)
    gate''', '''            HW=hw, HOIST=hoist, PREF=hoist, PV16=hw, num_warps=warps, num_stages=stages,
            **({"maxnreg": VERIFY_MAXNREG} if hw and _MAXNREG_OK and L != 1 and cfg["window"] else {}))
    gate''')
sub('# okc: problem=muse_attention_decode\n"""v20:', '# okc: problem=muse_attention_decode\n'
    '"""v21' + ("b" if occ == 2 else "") + ': v20b with sliding verify on 8-warp programs capped at 128 registers'
    + (' and ~2 programs per SM' if occ == 2 else '') + ' (tools/build_attn_v21.py).\n\nv20:')
open(DST, "w").write(s)
print("ok", DST)
