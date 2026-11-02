"""Offline NVRTC sm_90a compile of tools/bf16_v31_stream.cu with every instantiation the v31 runs files use
(no GPU needed): NVRTC version, per-kernel registers / spills / smem. Usage: <nvrtc venv>/bin/python
tools/tests/bf16_v31_nvrtc_cc.py [file.cu]"""
import re, sys, time
from cuda.bindings import nvrtc
sys.path.insert(0, "tools")
from bf16_v31_cfg import ALL_INSTANCES, name_expr
path = sys.argv[1] if len(sys.argv) > 1 else "tools/bf16_v31_stream.cu"
src = open(path).read().encode()
_, major, minor = nvrtc.nvrtcVersion()
t0 = time.time()
err, prog = nvrtc.nvrtcCreateProgram(src, b"bf16_v31.cu", 0, [], [])
names = [name_expr(c) for c in ALL_INSTANCES]
for nm in names:
    nvrtc.nvrtcAddNameExpression(prog, nm.encode())
opts = [b"-arch=sm_90a", b"-std=c++17", b"-Xptxas=-v"]
(res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
_, n = nvrtc.nvrtcGetProgramLogSize(prog)
log = b" " * n
nvrtc.nvrtcGetProgramLog(prog, log)
log = log.decode(errors="ignore")
print(f"NVRTC {major}.{minor}", res, f"{len(names)} instances, {time.time() - t0:.1f} s")
if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
    print(log[:4000]); sys.exit(1)
lowered = {}
for nm in names:
    _, low = nvrtc.nvrtcGetLoweredName(prog, nm.encode())
    lowered[low.decode()] = nm
cur = None
bad = 0
for l in log.split("\n"):
    m = re.search(r"Compiling entry function '(\S+)'", l)
    if m:
        cur = lowered.get(m.group(1), m.group(1))
    if "Used" in l and cur:
        print(f"  {cur:42s} {l.split('info    :')[-1].strip()}")
    if "spill" in l and cur and not re.search(r"0 bytes spill stores, 0 bytes spill loads", l):
        print(f"  {cur:42s} SPILL {l.strip()}"); bad += 1
print("spilling instances:", bad)
