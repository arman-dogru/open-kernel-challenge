"""Builds muse_prefill_layer/runs/v3_gs_after_local.py from v2b_dequant_grouped.py (exact replacements, asserted once;
the new expert path is tools/prefill_v3_experts.py.txt). Run at HEAD; the committed file must be byte-identical."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = [(ROOT / "muse_prefill_layer/runs/v2b_dequant_grouped.py").read_text()]
NEW = (ROOT / "tools/prefill_v3_experts.py.txt").read_text()


def rep(a, b):
    assert S[0].count(a) == 1, (a[:70], S[0].count(a))
    S[0] = S[0].replace(a, b)


rep('''# okc: problem=muse_prefill_layer
"""prefill v2b''', '''# okc: problem=muse_prefill_layer
"""prefill v3 = v2b with the review of #16:
  - m1: the NVFP4 FP32 tensor scale is applied in FP32 after each grouped GEMM (GEMM-1: in the SiLU kernel; GEMM-2:
    in the un-sort kernel) instead of being folded into the weights before their BF16 rounding. e2m1 x e4m3 is exact
    in BF16, so the dequantized weights are now exact and the only roundings are those of the activations, as on
    the Marlin path.
  - only local routes are gathered, activated and un-sorted (one program per row that returns early otherwise):
    v2b's full-capacity x[perm] and index_select moved all T x 8 rows (1.05 ms at 16K tokens).
  - torch._grouped_mm is guarded (attribute present and sm_90); otherwise Marlin at every token count.
  - buffers keep T x 8 rows (static under graph capture; all 8 of a token's experts can be local), see
    _experts_grouped.

prefill v2b''')
i = S[0].index("# Token counts at or above this run the experts as dequantized BF16 grouped GEMMs (v2); below, Marlin.")
j = S[0].index("def _experts(x, topk_weights, topk_ids, state):")
S[0] = S[0][:i] + NEW + S[0][j:]
rep('''        "side_stream": torch.cuda.Stream(device),''', '''        "side_stream": torch.cuda.Stream(device),
        "grouped_ok": _grouped_ok(device),''')
rep('''    experts_fn = _experts_grouped if tokens >= GROUPED_MIN_TOKENS else _experts''',
    '''    experts_fn = _experts_grouped if tokens >= GROUPED_MIN_TOKENS and state["grouped_ok"] else _experts''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_prefill_layer/runs/v3_gs_after_local.py"
dst.write_text(S[0])
