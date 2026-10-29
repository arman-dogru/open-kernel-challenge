import sys
from cuda.bindings import nvrtc
src = open(sys.argv[1]).read().encode()
for defs in [dict(NT=8, ST=4), dict(NT=16, ST=4), dict(NT=32, ST=4)]:
    err, prog = nvrtc.nvrtcCreateProgram(src, b"g.cu", 0, [], [])
    opts = [b"-arch=sm_90a", b"-std=c++17", b"-Xptxas=-v"] + [f"-D{k}={v}".encode() for k, v in defs.items()]
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog); log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log)
    lg = log.decode(errors="ignore")
    _, cs = nvrtc.nvrtcGetCUBINSize(prog)
    print(defs, r[0], "cubin", cs, [l.strip() for l in lg.split("\n") if "rror" in l or "Used" in l or "spill" in l or "arn" in l][:8])
