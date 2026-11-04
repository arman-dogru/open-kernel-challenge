"""v25e: NVRTC-compile every attn_fv / attn_fvws build the v25e diag times (the runs file's own _FV_SRC / _WS_SRC
and its _wsf / _ws_opts flag encoding). Non-ablation builds must have no C7515 and 0 spills (NPW 2 at 96 registers
spills and is reported, not asserted). Usage: python tools/tests/attn_v25e_variants_nvrtc.py runs/<file>.py"""
import ast, sys, hashlib
from cuda.bindings import nvrtc
t = ast.parse(open(sys.argv[1]).read())
srcs = {n.targets[0].id: n.value.value for n in t.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("_FV_SRC", "_WS_SRC")}
ns = {}
for n in t.body:
    if isinstance(n, ast.FunctionDef) and n.name in ("_wsf", "_ws_npw", "_ws_opts"):
        exec(compile(ast.Module([n], []), "x", "exec"), ns)
W = ns["_wsf"]
V = [(3, 0), (3, W(maxr=112, ws=0)), (3, 64), (2, 64)] + [(3, W(abl=a)) for a in (1, 2, 3, 32, 4, 8, 12, 16, 64)] + [
    (3, W(cbar=1)), (3, W(abl=1, cbar=1)), (3, W(npw=2, maxr=96)), (3, W(npw=2, cbar=1, maxr=96)),
    (3, W(npw=4, maxr=80, smr=104)), (3, W(npw=4, cbar=1, maxr=80, smr=104)), (2, W(npw=4, cbar=1, maxr=80, smr=104)),
    (3, W(abl=2, npw=4, maxr=80, smr=104)), (3, W(abl=1, npw=4, maxr=80, smr=104)), (6, W(npw=4, cbar=1, maxr=168)),
    (2, W(npw=4, cbar=1, maxr=168))]
for st, fl in V:
    name = "_WS_SRC" if fl & 64 else "_FV_SRC"
    err, prog = nvrtc.nvrtcCreateProgram(srcs[name].encode(), b"attn_fv.cu", 0, [], [])
    o = [b"-arch=sm_90a", b"-std=c++17", b"-Xptxas=-v", f"-DST={st}".encode()] + ns["_ws_opts"](fl)
    (res,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog); log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log); log = log.decode(errors="ignore")
    assert res == nvrtc.nvrtcResult.NVRTC_SUCCESS, log[:2000]
    abl = (fl >> 7) & 127 if fl & 64 else 0
    spill = " 0 bytes spill stores" not in log
    if not abl:
        assert "C7515" not in log, (st, fl, "C7515")
        assert not spill or b"-DNPW=2" in o, (st, fl, "spills")
    _, sz = nvrtc.nvrtcGetCUBINSize(prog); cub = b" " * sz; nvrtc.nvrtcGetCUBIN(prog, cub)
    regs = [l.split("Used")[1].split(",")[0].strip() for l in log.split("\n") if "Used" in l]
    print(f"{name} ST {st} flags {fl} {[x.decode() for x in o[3:]]}: {hashlib.sha256(cub).hexdigest()[:8]} {regs}"
          f"{' SPILLS' if spill else ''}{' C7515' if 'C7515' in log else ''}")
