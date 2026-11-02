"""Build muse_attention_decode/runs/v22_cuda_fv.py from v20b_verify_pref64.py + tools/attn_v22_fv.cu (byte-exact).

v22 = v20b with full-attention verify (q_len 8) at batch >= 2 (tp8) on a hand-written CUDA sm_90a split-KV kernel
(NVRTC in-process, cubin loaded from memory by CuPy, launched on torch's stream) instead of SGLang's FA3: those two
loads are ~75% of their load's time (weight 16 x ~200 / ~385 us) and ran at 1.03 / 1.02x. Kernel: tools/attn_v22_fv.cu
(FP8 wgmma Q.K^T and P.V, one warpgroup per KV head, V transposed per tile in shared memory, online softmax in
registers, 3-stage cp.async page-table gather); its partials go through the same Triton combine / RMSNorm x
sigmoid(gate) as v20b. The cubin is compiled (NVRTC) and loaded at import on an sm_90 device, never inside CUDA-graph
capture (review #20 m3); launches run under cp.cuda.Device(q's device) on that device's current torch stream.
"""
import sys
SRC, CU, DST = ("muse_attention_decode/runs/v20b_verify_pref64.py", "tools/attn_v22_fv.cu",
                "muse_attention_decode/runs/v22_cuda_fv.py")
s = open(SRC).read()
cu = open(CU).read()
assert '"""' not in cu and not cu.endswith("\\")  # embedded as a raw string


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


FV = '''_FV_SRC = r"""
''' + cu + '''"""
FV_BN, FV_ST = 64, 3  # must match BN / ST in tools/attn_v22_fv.cu
FV_SMEM = 2 * FV_ST * FV_BN * 128 + 2 * 2 * 9216 + 2 * 8192 + 1024  # K + V E4M3 stages, 2 V^T buffers (72 rows) and Q, alignment
FV_OCC = 4  # split-KV programs per SM for _nsplit (the kernel fits 2 per SM; the rest is load balance)
_FV = {}  # "cubin": NVRTC output (device-independent); device index -> (module, function); "sm90": index -> bool


def _fv_cubin():
    """NVRTC (cuda.bindings, in-process) -> sm_90a cubin bytes (no files). Host-only, no CUDA calls."""
    if "cubin" not in _FV:
        from cuda.bindings import nvrtc
        err, prog = nvrtc.nvrtcCreateProgram(_FV_SRC.encode(), b"attn_fv.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        opts = [b"-arch=sm_90a", b"-std=c++17"]
        (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="ignore"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        _FV["cubin"] = cubin
    return _FV["cubin"]


def _fv_fn(dev):
    """The kernel loaded on CUDA device dev by CuPy from memory. Loaded at import (below) for the current device,
    so the NVRTC compile and module load never run inside the judge's CUDA-graph capture."""
    if dev not in _FV:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("attn_fv: kernel not loaded before CUDA-graph capture (device %d)" % dev)
        import cupy as cp
        cubin = _fv_cubin()
        with cp.cuda.Device(dev):
            mod = cp.cuda.function.Module()
            mod.load(cubin)
            f = mod.get_function("attn_fv")
            drv = cp.cuda.driver
            drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8), FV_SMEM)
        _FV[dev] = (mod, f)
    return _FV[dev][1]


def _fv_sm90(device):
    """True on an sm_90 (H100 / H200) device: the cubin is sm_90a only."""
    key = ("sm90", device.index)
    if key not in _FV:
        _FV[key] = torch.cuda.get_device_capability(device) == (9, 0)
    return _FV[key]


if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
    _fv_fn(torch.cuda.current_device())  # compile + load at import (review #20 m3), not on the first call


def _fv_ok(kind, L, batch, hq, hkv, d, q):
    return (kind == "full" and L == 8 and batch >= 2 and (hq, hkv, d) == (16, 2, 64) and q.device.type == "cuda"
            and _fv_sm90(q.device))


def _fv_page_ok(page):
    return page & (page - 1) == 0  # the kernel indexes pages by shift


def _fv(data, q, qkv, cfg):
    """Full verify at batch >= 2: CUDA split-KV partials (tools/attn_v22_fv.cu), then the Triton combine."""
    import cupy as cp
    import numpy as np
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = batch * L
    page = cfg["page_size"]
    capacity = data["page_table"].shape[1] * page
    nsplit = _nsplit(cfg, batch, 1, capacity, FV_OCC)
    chunk = triton.cdiv(capacity, nsplit)
    o_part = torch.empty((T * hq * nsplit, d), device=q.device, dtype=torch.float32)
    lse = torch.empty((T * hq * nsplit,), device=q.device, dtype=torch.float32)
    pt, sl = data["page_table"], data["cache_seqlens"]
    assert pt.dtype == torch.int32 and sl.dtype == torch.int32 and q.is_contiguous()
    kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    f = _fv_fn(dev)
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(q.device).cuda_stream):
        f((nsplit, batch), (256,), (
            np.uint64(q.data_ptr()), np.int32(q.stride(0)), np.uint64(kc.data_ptr()), np.uint64(vc.data_ptr()),
            np.uint64(pt.data_ptr()), np.int32(pt.stride(0)), np.uint64(sl.data_ptr()),
            np.uint64(data["k_scale"].data_ptr()), np.uint64(data["v_scale"].data_ptr()),
            np.uint64(o_part.data_ptr()), np.uint64(lse.data_ptr()),
            np.float32(cfg["sm_scale"] * cfg["qk_scale_factor"]), np.int32(chunk), np.int32(page.bit_length() - 1), np.int32(nsplit),
            np.int32(hq)), shared_mem=FV_SMEM)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = torch.empty((T, hq * d), device=qkv.device, dtype=data["qkv"].dtype)
    _combine_norm_gate[(T, hq)](o_part, lse, gate, qkv.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                num_warps=4 if nsplit >= 64 else 1)
    return out.view(batch, L, hq * d)


def kernel_fn(data):'''
sub("def kernel_fn(data):", FV)
sub('''    if path is None and q.device.type == "cpu":''', '''    if path is None and _fv_ok(kind, L, batch, hq, hkv, d, q) and _fv_page_ok(cfg["page_size"]):
        return _fv(data, q, qkv, cfg)  # v22: CUDA full-verify kernel instead of FA3
    if path is None and q.device.type == "cpu":''')
sub('# okc: problem=muse_attention_decode\n"""v20:', '# okc: problem=muse_attention_decode\n'
    '"""v22: v20b with full-attention verify at batch >= 2 on a CUDA sm_90a split-KV kernel (FP8 wgmma,\n'
    'cp.async page-table gather) instead of FA3. See tools/build_attn_v22.py and tools/attn_v22_fv.cu.\n\nv20:')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
