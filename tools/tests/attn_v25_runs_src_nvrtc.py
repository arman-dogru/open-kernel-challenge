import ast, sys, hashlib
from cuda.bindings import nvrtc
t = ast.parse(open(sys.argv[1]).read())
srcs = {n.targets[0].id: n.value.value for n in t.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("_FV_SRC", "_WS_SRC")}
for name, opts in (("_FV_SRC", [b"-DST=3"]), ("_WS_SRC", [b"-DST=3"]), ("_WS_SRC", [b"-DST=2"])):
    err, prog = nvrtc.nvrtcCreateProgram(srcs[name].encode(), b"attn_fv.cu", 0, [], [])
    o = [b"-arch=sm_90a", b"-std=c++17", b"-Xptxas=-v"] + opts
    (res,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog); log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log); log = log.decode(errors="ignore")
    assert res == nvrtc.nvrtcResult.NVRTC_SUCCESS, log[:2000]
    _, sz = nvrtc.nvrtcGetCUBINSize(prog); cub = b" " * sz; nvrtc.nvrtcGetCUBIN(prog, cub)
    print(name, opts, hashlib.sha256(cub).hexdigest()[:8], [l.split(":")[-1].strip() for l in log.split("\n") if "Used" in l or "spill" in l])
