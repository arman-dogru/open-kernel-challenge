"""Offline NVRTC sm_90a compile of tools/moe_v27_stream.cu for every config the v27 runs files use (no GPU):
registers / spills / smem per (GATED, NT, NW, ST, TPS, FASTSCALE). Usage: <nvrtc venv>/bin/python tools/tests/moe_v27_nvrtc_cc.py"""
import itertools, re, sys, time
from cuda.bindings import nvrtc
src = open(sys.argv[1] if len(sys.argv) > 1 else "tools/moe_v27_stream.cu").read().encode()
sys.path.insert(0, "tools")
from moe_v27_cfg import ALL_CONFIGS
bad = 0
t0 = time.time()
for (gated, nt, nw, st, tps), fast in itertools.product(ALL_CONFIGS, (1, 0)):
    err, prog = nvrtc.nvrtcCreateProgram(src, b"moe_v27.cu", 0, [], [])
    opts = [b"-arch=sm_90a", b"-std=c++17", b"-Xptxas=-v"] + [f"-D{k}={v}".encode() for k, v in
            dict(GATED=gated, NT=nt, NW=nw, ST=st, TPS=tps, FASTSCALE=fast).items()]
    (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    log = log.decode(errors="ignore")
    if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        print("FAIL", gated, nt, nw, st, tps, fast, log[:3000]); bad += 1; continue
    used = [l.split("info    :")[-1].strip() for l in log.split("\n") if "Used" in l]
    spill = [l for l in log.split("\n") if "spill" in l and "0 bytes spill stores, 0 bytes spill loads" not in l]
    print(f"GATED {gated} NT {nt} NW {nw} ST {st} TPS {tps} FAST {fast}: {used[0] if used else '?'}{' SPILL ' + spill[0].strip() if spill else ''}")
    bad += bool(spill)
print(f"{time.time() - t0:.1f} s, bad {bad}")
