"""Builds muse_prefill_layer/runs/v1_router_fused.py from muse_prefill_layer/baseline.py (SGLang's layer), with
exact replacements asserted once each. Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_prefill_layer/baseline.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""SGLang\'s decoder layer', '''# okc: problem=muse_prefill_layer
"""prefill v1 = SGLang's layer (baseline.py) with the router's two split-weight BF16 GEMMs (hi and lo halves of the FP32
router, each [T, 8192] x [8192, 256] with FP32 output) run as one GEMM against [hi | lo] ([8192, 512]), then the two
halves summed in FP32. Each half is the same product as before, and the sum is the same FP32 add. It reads the
[T, 8192] BF16 FFN input once instead of twice (268 MB at 16K tokens) and drops a launch on the side stream that
overlaps the pooler. Everything else is SGLang's, unchanged.

SGLang's decoder layer''')
rep('''        "router_lo_t": lo.t().contiguous(),''', '''        "router_lo_t": lo.t().contiguous(),
        "router_hilo_t": torch.cat((hi, lo), dim=0).t().contiguous(),  # [8192, 512]: hi columns 0-255, lo 256-511''')
rep('''            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)''',
'''            hilo = torch.mm(ffn_input, state["router_hilo_t"], out_dtype=torch.float32)
            logits = hilo[:, :N_EXPERTS] + hilo[:, N_EXPERTS:]''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_prefill_layer/runs/v1_router_fused.py"
dst.write_text(s)
