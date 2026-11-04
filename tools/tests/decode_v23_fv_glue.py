"""Decode v23: the full-verify CUDA path's host glue in muse_decode_layer/runs/v23_fv_attn.py, on CPU (no GPU).

The decode file can't be imported on CPU as-is (SGLang / sgl_kernel / CuPy), so those modules are stubbed; nothing
from them runs here. The CUDA launch is replaced by a fake kernel that checks every launch argument against the
expected tensors / values (pointer by pointer, grid, block, shared memory) and fills the partials with attention's
step-by-step emulation of tools/attn_v23_fv.cu (tools/tests/attn_v22_fv_kvscale.py, same math). The decode file's
own _fv + Triton combine (TRITON_INTERPRET=1) then produce the output, which is checked against attention's
reference (muse_attention_decode/reference.py: the same attention sub-problem). Also checks that _FV_SRC is
byte-identical to tools/attn_v23_fv.cu and that kernel_fn dispatches full verify at batch >= 2 to _fv.
Usage: /workspace/okc-cpu-venv/bin/python tools/tests/decode_v23_fv_glue.py [decode runs file]"""
import ctypes, importlib.util, math, os, sys, types
os.environ.setdefault("TRITON_INTERPRET", "1")
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "muse_decode_layer", "runs", "v23_fv_attn.py")
import torch


class _Any(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, *a, **k: None})


for m in ["sgl_kernel", "sgl_kernel.flash_attn", "sgl_kernel.scalar_type", "sglang", "sglang.kernels",
          "sglang.kernels.ops", "sglang.kernels.ops.activation", "sglang.kernels.ops.activation.activation",
          "sglang.kernels.ops.moe", "sglang.kernels.ops.moe.moe_wna16_marlin", "sglang.kernels.ops.muse_spark_v12",
          "sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing", "sglang.kernels.ops.muse_spark_v12.fused_norm_gate",
          "sglang.kernels.ops.muse_spark_v12.fused_norm_residual",
          "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton", "sglang.srt",
          "sglang.srt.layers", "sglang.srt.layers.moe", "sglang.srt.layers.moe.fused_moe_triton",
          "sglang.srt.layers.muse_spark_layers", "sglang.srt.layers.quantization",
          "sglang.srt.layers.quantization.marlin_utils", "sglang.srt.layers.quantization.marlin_utils_fp4",
          "sglang.srt.layers.quantization.unquant"]:
    sys.modules[m] = _Any(m)


class _Ctx:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


cp = types.ModuleType("cupy")
cp.cuda = types.SimpleNamespace(Device=_Ctx, ExternalStream=_Ctx)
sys.modules["cupy"] = cp
spec = importlib.util.spec_from_file_location("dec", path)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

# Attention's reference and its v23 emulation (same attention sub-problem as the decode layer's).
sys.path.insert(0, os.path.join(root, "muse_attention_decode"))
sys.argv = [sys.argv[0], os.path.join(root, "muse_attention_decode", "runs", "v23_fv_addr.py")]
tspec = importlib.util.spec_from_file_location("kvtest", os.path.join(root, "tools", "tests", "attn_v22_fv_kvscale.py"))
KT = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(KT)
R, A = KT.R, KT.S
E4 = torch.float8_e4m3fn


def _view(ptr, n, dtype):
    ct = {torch.float32: ctypes.c_float}[dtype]
    return torch.frombuffer((ct * n).from_address(ptr), dtype=dtype, count=n)


def run(kw, vs):
    data = KT._make(kw, vs, "cpu")
    ref_out = KT.emulate(data)  # attention's emulation + attention's combine (also writes the new cache rows)
    cfg = data["config"]
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    T = batch * L
    qkv = data["qkv"].view(T, -1)
    q = A._norm_rope_store_fused(data, qkv.float(), cfg, torch.float32).view(T, -1)[:, : hq * d].to(E4).contiguous()
    # emulate() recomputes q the same way; cache rows already replaced, so a second call is idempotent.
    gate = qkv[:, (hq + 2 * hkv) * d:].float()  # the decode file's gate slice (interpreter: no bf16)
    page = cfg["page_size"]
    capacity = data["page_table"].shape[1] * page
    nsplit = A._nsplit(cfg, batch, 1, capacity, A.FV_OCC)
    seen = {}

    def fake_f(grid, block, args, shared_mem):
        assert grid == (nsplit, batch) and block == (256,) and shared_mem == D.FV_SMEM == A.FV_SMEM
        kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
        exp = [q.data_ptr(), q.stride(0), kc.data_ptr(), vc.data_ptr(), data["page_table"].data_ptr(),
               data["page_table"].stride(0), data["cache_seqlens"].data_ptr(), data["k_scale"].data_ptr(),
               data["v_scale"].data_ptr(), None, None, cfg["sm_scale"] * cfg["qk_scale_factor"],
               -(-capacity // nsplit), page.bit_length() - 1, nsplit, hq]
        assert len(args) == len(exp)
        for i, (a, e) in enumerate(zip(args, exp)):
            if e is not None:
                assert abs(float(a) - float(e)) <= 1e-6 * abs(float(e)), (i, a, e)
        # Partials: rerun attention's emulation loop by temporarily capturing its o_part / lse.
        cap = {}
        orig = A._combine_norm_gate
        class Grab:
            def __getitem__(self, g):
                def f(o_part, lse, *r, **k):
                    cap["o"], cap["l"] = o_part, lse
                return f
        A._combine_norm_gate = Grab()
        try:
            KT.emulate(data)
        finally:
            A._combine_norm_gate = orig
        n = T * hq * nsplit
        _view(int(args[9]), n * d, torch.float32).copy_(cap["o"].reshape(-1))
        _view(int(args[10]), n, torch.float32).copy_(cap["l"])
        seen["launched"] = True

    D._fv_fn = lambda dev: fake_f
    D.torch.cuda.current_stream = lambda dev=None: types.SimpleNamespace(cuda_stream=0)
    D.torch.cuda.current_device = lambda: 0
    out = D._fv(data, q, gate, cfg)
    assert seen.get("launched") and out.shape == (T, hq * d) and out.dtype == torch.bfloat16
    diff = (out.float() - ref_out.reshape(T, -1).to(torch.bfloat16).float()).abs().max().item()
    res = R.check_implementation(data, R.collect(data, out.view(batch, L, hq * d)))
    print("glue", kw, "max |decode - attention| %.3g" % diff, res, flush=True)
    return bool(res[0]) and diff < 2e-2


def main():
    s = open(path).read()
    cu = open(os.path.join(root, "tools", "attn_v23_fv.cu")).read()
    a = s.index('_FV_SRC = r"""\n') + len('_FV_SRC = r"""\n')
    ok = s[a:s.index('"""', a)] == cu
    print("_FV_SRC == tools/attn_v23_fv.cu:", ok)
    # dispatch: full verify b >= 2 goes to _fv (on an sm_90 CUDA q), everything else as v19
    fake_q = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=0))
    D._FV[("sm90", 0)] = True
    for kind, L, b, want in [("full", 8, 16, True), ("full", 8, 32, True), ("full", 8, 2, True), ("full", 8, 1, False),
                             ("sliding", 8, 16, False), ("full", 1, 16, False)]:
        took = D._attn_path(kind, L, b, 2, 64) is None and D._fv_ok(kind, L, b, 16, 2, 64, fake_q)
        print("dispatch", kind, "q_len", L, "batch", b, "->", "fv" if took else "v19 path"); ok &= took == want
    for kw, vs in KT.CASES:  # kv_scale != 1 (k / v scales differ in two), pages 1 / default / 16
        ok &= run(kw, vs)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
