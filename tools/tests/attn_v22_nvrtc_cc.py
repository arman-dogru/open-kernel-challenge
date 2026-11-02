"""Offline NVRTC sm_90a compile of tools/attn_v22_fv.cu (no GPU needed); prints NVRTC version, regs and spills.
Usage: <venv with cuda-bindings + nvrtc>/bin/python tools/tests/attn_v22_nvrtc_cc.py [file.cu]
Registers depend on the NVRTC/ptxas version, so always quote the version printed here with the figure."""
import sys
from cuda.bindings import nvrtc
path = sys.argv[1] if len(sys.argv) > 1 else "tools/attn_v22_fv.cu"
src = open(path).read().encode()
_, major, minor = nvrtc.nvrtcVersion()
err, prog = nvrtc.nvrtcCreateProgram(src, b"attn_fv.cu", 0, [], [])
opts = [b"-arch=sm_90a", b"-std=c++17", b"-Xptxas=-v"]  # the kernel's own options plus -v
(res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
_, n = nvrtc.nvrtcGetProgramLogSize(prog)
log = b" " * n
nvrtc.nvrtcGetProgramLog(prog, log)
lines = [l.strip() for l in log.decode(errors="ignore").split("\n") if l.strip()]
_, cs = nvrtc.nvrtcGetCUBINSize(prog)
print(f"NVRTC {major}.{minor}", res, "cubin", cs)
for l in lines:
    if "rror" in l or "arn" in l or "Used" in l or "spill" in l:
        print("  " + l)
sys.exit(0 if res == nvrtc.nvrtcResult.NVRTC_SUCCESS else 1)
