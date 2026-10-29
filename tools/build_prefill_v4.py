"""Builds muse_prefill_layer/runs/v4_dequant_rows4.py from v3_gs_after_local.py (exact replacements, asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = [(ROOT / "muse_prefill_layer/runs/v3_gs_after_local.py").read_text()]


def rep(a, b):
    assert S[0].count(a) == 1, (a[:70], S[0].count(a))
    S[0] = S[0].replace(a, b)


rep('''# okc: problem=muse_prefill_layer
"""prefill v3''', '''# okc: problem=muse_prefill_layer
"""prefill v4 = v3 with the NVFP4 dequant on 4 weight rows per program (a [4, 2048]-byte block: 65K programs instead
of 262K per w13 call, 4x the bytes in flight per program). It took 1.6 ms (two calls, ~4 GB moved, ~2.5 TB/s) at
16K tokens in v2b / v3. Same values (exact e2m1 x e4m3 in BF16).

prefill v3''')
rep('''def _nvfp4_dequant_kernel(q_ptr, sf_ptr, out_ptr, ROW_BYTES: tl.constexpr):''',
    '''def _nvfp4_dequant_kernel(q_ptr, sf_ptr, out_ptr, ROW_BYTES: tl.constexpr, ROWS: tl.constexpr):''')
rep('''    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, ROW_BYTES)
    b = tl.load(q_ptr + row * ROW_BYTES + c).to(tl.int32)
    sfb = tl.load(sf_ptr + row * (ROW_BYTES // 8) + c // 8)  # e4m3 scale bytes as uint8''',
'''    row = (tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS))[:, None]  # ROWS rows per program (v4)
    c = tl.arange(0, ROW_BYTES)[None, :]
    b = tl.load(q_ptr + row * ROW_BYTES + c).to(tl.int32)
    sfb = tl.load(sf_ptr + row * (ROW_BYTES // 8) + c // 8)  # e4m3 scale bytes as uint8''')
rep('''    v = tl.reshape(tl.join(vlo, vhi), (2 * ROW_BYTES,))  # element 2i = low nibble, 2i + 1 = high
    tl.store(out_ptr + row * (2 * ROW_BYTES) + tl.arange(0, 2 * ROW_BYTES), v.to(tl.bfloat16))''',
'''    v = tl.reshape(tl.join(vlo, vhi), (ROWS, 2 * ROW_BYTES))  # element 2i = low nibble, 2i + 1 = high
    tl.store(out_ptr + row * (2 * ROW_BYTES) + tl.arange(0, 2 * ROW_BYTES)[None, :], v.to(tl.bfloat16))''')
rep('''    _nvfp4_dequant_kernel[(e * n,)](q, sf.view(torch.uint8), out, ROW_BYTES=kb, num_warps=8)''',
    '''    assert (e * n) % 4 == 0
    _nvfp4_dequant_kernel[(e * n // 4,)](q, sf.view(torch.uint8), out, ROW_BYTES=kb, ROWS=4, num_warps=8)''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_prefill_layer/runs/v4_dequant_rows4.py"
dst.write_text(S[0])
