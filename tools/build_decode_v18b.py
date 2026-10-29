"""Builds muse_decode_layer/runs/v18b_cuda_experts_192.py from v18_cuda_experts_256.py (exact replacements, each
asserted once). Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v18_cuda_experts_256.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''# okc: problem=muse_decode_layer
"""v18:''', '''# okc: problem=muse_decode_layer
"""v18b: v18 with the CUDA experts capped at 192 tokens (128-token verify 16 keeps NB 2; 256-token verify 32 back on
Marlin). v18's check 9566de62f7e94ed1 (1.333x): NB 2 at 128 tokens beat Marlin (verify 16 sliding 422 -> 405 us, full
619 -> 587 us vs v17) while NB 4 at 256 lost (verify 32 sliding 600 -> 673, full 971 -> 1025 us: 115 registers,
2 CTAs/SM).

v18:''')
rep('''             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2)), 256: (4, (8, 8, 2), (8, 8, 2))}''',
    '''             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2))}''')
rep("LAYER_SMALL_MAX = 256  # tokens up to this use the CUDA experts (v17: <= 32; v18: <= 256, see docstring)",
    "LAYER_SMALL_MAX = 192  # tokens up to this use the CUDA experts (v17: <= 32; v18b: <= 192, NB 2 above 64)")
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v18b_cuda_experts_192.py"
dst.write_text(s)
