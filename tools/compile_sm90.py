"""Compile our Triton kernels for sm_90 on the CPU-only box (no GPU needed), to catch Triton
compile errors before spending an okc check. Usage: python tools/compile_sm90.py <problem>.
The box's Triton may be newer than the judge image's, so this is a strong hint, not proof."""
import os, sys, warnings
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
warnings.filterwarnings("ignore")
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(fn, types, const, warps=4, stages=3):
    sig = dict(types)
    for k in const:
        sig[k] = "constexpr"
    src = ASTSource(fn=fn, signature=sig, constexprs={(fn.arg_names.index(k),): v for k, v in const.items()})
    triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": warps, "num_stages": stages})


def attn(S):
    t = {"q_ptr": "*bf16", "stride_qt": "i32", "k_ptr": "*u8", "v_ptr": "*u8", "pt_ptr": "*i32", "stride_pt": "i32",
         "seqlens_ptr": "*i32", "ks_ptr": "*fp32", "vs_ptr": "*fp32", "o_ptr": "*fp32", "lse_ptr": "*fp32",
         "qk_scale": "fp32", "chunk": "i32", "PAGE": "i32"}
    for L in (1, 8):
        for W in (0, 2048):
            build(S._attn_split, t, dict(L=L, G=8, HKV=2, NSPLIT=8, WINDOW=W, BLOCK_M=max(16, 8 * L), BLOCK_N=64, D=64, HW=True, HOIST=False))
        for hoist in (False, True):
            build(S._attn_split, t, dict(L=L, G=8, HKV=2, NSPLIT=64, WINDOW=0, BLOCK_M=max(16, 8 * L), BLOCK_N=128,
                                         D=64, HW=True, HOIST=hoist), stages=4)
            build(S._attn_split_rows, t, dict(L=L, G=8, HKV=2, NSPLIT=8, WINDOW=0, BLOCK_M=16 * L, BLOCK_N=64, D=64,
                                              HW=True, HOIST=hoist), warps=4 if L == 1 else 8)
    build(S._combine_norm_gate, {"o_ptr": "*fp32", "lse_ptr": "*fp32", "gate_ptr": "*bf16", "stride_gt": "i32",
                                 "out_ptr": "*bf16", "eps": "fp32"}, dict(HQ=16, NSPLIT=8, D=64), warps=1)


def bf16(S):
    t = {"x_ptr": "*bf16", "w_ptr": "*bf16", "lo_ptr": "*bf16", "out_ptr": "*bf16", "M": "i32", "N": "i32", "K": "i32",
         "stride_xm": "i32", "stride_wn": "i32", "stride_om": "i32"}
    build(S._swapab_gemm, t, dict(BM=16, BN=64, BK=128, SPLIT_K=1, HAS_LO=False, OUT_BF16=True), stages=4)
    t2 = dict(t, out_ptr="*fp32")
    build(S._swapab_gemm, t2, dict(BM=16, BN=16, BK=256, SPLIT_K=8, HAS_LO=True, OUT_BF16=False), stages=4)


def moe(S):
    t = {"a_ptr": "*bf16", "w_ptr": "*u8", "sf_ptr": "*u8", "gs_ptr": "*fp32", "out_ptr": "*bf16", "sorted_ptr": "*i32",
         "eids_ptr": "*i32", "npad_ptr": "*i32", "numel": "i32", "local_start": "i32", "N": "i32", "K": "i32"}
    build(S._moe_gemm, t, dict(A_ROW_DIV=8, GATED=True, N_LOCAL=32, BM=16, BN=64, BK=128))


if __name__ == "__main__":
    prob = sys.argv[1]
    sys.path.insert(0, os.path.join(ROOT, prob))
    import submission as S
    {"muse_attention_decode": attn, "muse_bf16_proj": bf16, "muse_moe_w4a16": moe}[prob](S)
    print(prob, "all kernels compiled for sm_90")
