"""Builds muse_bf16_proj/runs/v28_wgmma_pipe.py from v17_best_route.py + tools/bf16_v28_wgmma.cu (exact replacements,
each asserted once). Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_bf16_proj/runs/v17_best_route.py").read_text()
cu = (ROOT / "tools/bf16_v28_wgmma.cu").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""v17: v14 with', '''"""v28 (exploration): v27 fully software-pipelined: each CTA's k range is NCH = 4 64-k chunks, each with its own
W stage (no slot reuse), all NCH TMA loads issued up front, wgmma over the chunks with one wait_group at the end
(v27 drained every chunk), split-K raised so NCH = 4 everywhere (~512-2048 CTAs, 36-48 KB smem -> 4-6 CTAs/SM).

v27 (exploration): v17 plus an SM90a wgmma swap-AB GEMM (_WG_SRC; m64nNk16, N = tokens padded to 8 / 16 / 32,
FP32 accumulate) for qkv_gate, o, pooler and unpooler at 8 <= m <= 32 (v17's routes elsewhere). The weights are
repacked in prepare() (a pure permutation) into the no-swizzle K-major core-matrix image wgmma reads from shared
memory, streamed by 1-D cp.async.bulk (TMA) into 4 mbarrier stages; x is staged once per CTA in the same layout;
split-K partials are FP32 atomics, the last CTA of a 64-row tile writes BF16. Compiled in prepare() with NVRTC
(cuda.bindings, in-process, -arch=sm_90a) to a cubin, loaded from memory by CuPy (cupy.cuda.function.Module).

v17: v14 with''')
rep("_MODS = {}\n", '''_MODS = {}

_WG_SRC = r"""
''' + cu + '''"""
_WG = {}
# layer -> split S (64-k chunks per CTA = K / 64 / S): ~512 CTAs of one warpgroup each.
WG_SPLIT = {"qkv_gate": 32, "o": 4, "pooler": 32, "unpooler": 16, "unpooler_s": 16}  # K / 64 / S == WG_ST
WG_MIN_M, WG_MAX_M, WG_ST = 8, 32, 4


def _wg_mod(NT):
    """NVRTC (cuda.bindings, in-process) -> cubin for sm_90a (wgmma needs the arch-specific target) -> CuPy."""
    if NT not in _WG:
        import cupy as cp
        from cuda.bindings import nvrtc
        err, prog = nvrtc.nvrtcCreateProgram(_WG_SRC.encode(), b"wg.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        opts = [b"-arch=sm_90a", b"-std=c++17", f"-DNT={NT}".encode(), f"-DST={WG_ST}".encode()]
        (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="ignore"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        mod = cp.cuda.function.Module()
        mod.load(cubin)  # from memory: no files
        _WG[NT] = (mod, mod.get_function("gemm_wg"))
    return _WG[NT][1]


def _wg_repack(w):
    """[N, K] bf16 -> per (64-row tile r, 64-k chunk c) an 8 KB image [u 4][kg 2][rg 8][row 8][e 8] (a permutation)."""
    n, k = w.shape
    v = w.contiguous().view(n // 64, 8, 8, k // 64, 4, 2, 8)
    return v.permute(0, 3, 4, 5, 1, 2, 6).contiguous().view(-1)


def _wg_gemm(x, layer, n, state):
    import cupy as cp
    m, k = x.shape
    x = x.contiguous()
    assert x.data_ptr() % 16 == 0
    nt = 8 if m <= 8 else (16 if m <= 16 else 32)
    S = WG_SPLIT["unpooler_s" if (layer == "unpooler" and n == 1024) else layer]
    nch = k // 64 // S
    assert nch == WG_ST, (layer, k, S)
    f = _wg_mod(nt)
    smem = WG_ST * 8192 + nch * nt * 128
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        f((n // 64, S), (128,), (
            np.uint64(state[layer + "_wg"].data_ptr()), np.uint64(x.data_ptr()), np.uint64(out.data_ptr()),
            np.uint64(state[layer + "_wgws"].data_ptr()), np.uint64(state[layer + "_wgcnt"].data_ptr()),
            np.int32(m), np.int32(n), np.int32(k), np.int32(nch), np.int32(S)), shared_mem=smem)
    return out
''')
rep('''    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
''', '''    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
    if layer != "router" and WG_MIN_M <= m <= WG_MAX_M and x.is_cuda:
        return _wg_gemm(x, layer, n, state)
''')
rep('''            state[name + "_cnt"] = torch.zeros((n // 16,), device=weights[name].device, dtype=torch.int32)
''', '''            state[name + "_cnt"] = torch.zeros((n // 16,), device=weights[name].device, dtype=torch.int32)
            state[name + "_wg"] = _wg_repack(weights[name])
            state[name + "_wgws"] = torch.zeros((WG_MAX_M, n), device=weights[name].device, dtype=torch.float32)
            state[name + "_wgcnt"] = torch.zeros((n // 64,), device=weights[name].device, dtype=torch.int32)
        import cupy as cp
        drv = cp.cuda.driver
        attr = getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8)
        for nt in (8, 16, 32):  # NVRTC-compile now; allow up to 160 KB of dynamic shared memory
            drv.funcSetAttribute(_wg_mod(nt).ptr, attr, 160 * 1024)
        torch.cuda.synchronize()
''')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_bf16_proj/runs/v28_wgmma_pipe.py"
out.write_text(s)
