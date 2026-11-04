import ast, sys
from cuda.bindings import nvrtc
t = ast.parse(open(sys.argv[1]).read())
src = [n.value.value for n in t.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_CU_SRC"][0]
W,KS,G,PF,NB,F = map(int, sys.argv[2:8])
err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"moe.cu", 0, [], [])
o = [b"-arch=sm_90a", b"-std=c++17"] + [f"-D{k}={v}".encode() for k, v in (("WARPS",W),("KS",KS),("GATED",G),("PF",PF),("NB",NB),("FASTSCALE",F))]
(res,) = nvrtc.nvrtcCompileProgram(prog, len(o), o); assert res == nvrtc.nvrtcResult.NVRTC_SUCCESS
_, sz = nvrtc.nvrtcGetCUBINSize(prog); cub = b" " * sz; nvrtc.nvrtcGetCUBIN(prog, cub)
open(sys.argv[8], "wb").write(cub)
