"""Build muse_attention_decode/runs/v22_cuda_fv.py from v20b_verify_pref64.py + tools/attn_v22_fv.cu (byte-exact).

v22 = v20b with full-attention verify (q_len 8) at batch >= 2 (tp8) on a hand-written CUDA sm_90a split-KV kernel
(NVRTC in-process, cubin loaded from memory by CuPy, launched on torch's stream) instead of SGLang's FA3: those two
loads are ~75% of their load's time (weight 16 x ~200 / ~385 us) and ran at 1.03 / 1.02x. Kernel: tools/attn_v22_fv.cu
(FP8 wgmma Q.K^T and P.V, one warpgroup per KV head, V transposed per tile in shared memory, online softmax in
registers, 3-stage
cp.async page-table gather); its partials go through the same Triton combine / RMSNorm x sigmoid(gate) as v20b.
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
FV_SMEM = 2 * FV_ST * FV_BN * 128 + 2 * 2 * 8192 + 2 * 8192 + 1024  # K + V E4M3 stages, 2 V^T buffers and Q (2 heads x 64 x 128 B), alignment
FV_OCC = 4  # split-KV programs per SM for _nsplit (the kernel fits 2 per SM; the rest is load balance)
_FV = {}


def _fv_fn():
    """NVRTC (cuda.bindings, in-process) -> sm_90a cubin -> CuPy, loaded from memory (no files)."""
    if "f" not in _FV:
        import cupy as cp
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
        mod = cp.cuda.function.Module()
        mod.load(cubin)
        f = mod.get_function("attn_fv")
        drv = cp.cuda.driver
        drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8), FV_SMEM)
        _FV["m"], _FV["f"] = mod, f
    return _FV["f"]


def _fv_ok(kind, L, batch, hq, hkv, d, q):
    return kind == "full" and L == 8 and batch >= 2 and (hq, hkv, d) == (16, 2, 64) and q.device.type == "cuda"


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
    f = _fv_fn()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
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
