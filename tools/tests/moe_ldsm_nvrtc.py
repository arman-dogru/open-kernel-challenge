"""NVRTC-compile moe_mma for every SMALL_CFG (WARPS, KS, GATED, PF, NB) of a runs file, FASTSCALE 0/1, and print
registers / spills; with a second file, compare. Usage: python tools/tests/moe_ldsm_nvrtc.py runs/a.py [runs/b.py]"""
import ast, sys, re
from cuda.bindings import nvrtc


def load(path):
    t = ast.parse(open(path).read())
    src = cfg = None; H = I = 4096
    for n in t.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id == "_CU_SRC": src = n.value.value
            if n.targets[0].id == "SMALL_CFG": cfg = ast.literal_eval(n.value)
    return src, cfg


def comp(src, opts):
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"moe.cu", 0, [], [])
    o = [b"-arch=sm_90a", b"-std=c++17", b"-Xptxas=-v"] + [x.encode() for x in opts]
    (res,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog); log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log); log = log.decode(errors="ignore")
    assert res == nvrtc.nvrtcResult.NVRTC_SUCCESS, log[:3000]
    out = {}
    cur = None
    for l in log.split("\n"):
        m = re.search(r"Compiling entry function '(\w+)'", l)
        if m: cur = m.group(1)
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", l)
        if m and cur: out.setdefault(cur, {})["spill"] = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"Used (\d+) registers", l)
        if m and cur: out.setdefault(cur, {})["regs"] = int(m.group(1))
    return out.get("moe_mma")


res = []
for path in sys.argv[1:]:
    src, cfg = load(path)
    r = {}
    for bs, (NB, g1, g2) in sorted(cfg.items()):
        for gated, (W, S, P) in ((1, g1), (0, g2)):
            for fast in (0, 1):
                key = (W, 4096 // S, gated, P, NB, fast)
                if key not in r:
                    r[key] = comp(src, [f"-DWARPS={W}", f"-DKS={4096 // S}", f"-DGATED={gated}", f"-DPF={P}", f"-DNB={NB}", f"-DFASTSCALE={fast}"])
    res.append(r)
    for k, v in r.items(): print(path.split("/")[-1], "WARPS KS GATED PF NB FAST", k, v)
if len(res) == 2:
    for k in res[0]:
        if k in res[1]: print("cmp", k, res[0][k], "->", res[1][k])
