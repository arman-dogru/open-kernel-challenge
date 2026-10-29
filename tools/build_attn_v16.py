"""Builds muse_attention_decode/runs/v16_verify_fused.py from v15b_verify_route.py (exact replacements, each asserted
once; the new kernel is tools/attn_verify_kernel.py.txt). Run at HEAD; the committed file must be byte-identical."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_attention_decode/runs/v15b_verify_route.py").read_text()
KERNEL = (ROOT / "tools/attn_verify_kernel.py.txt").read_text()
VERIFY_CFG = "(64, 4, 2, 4)"  # BLOCK_N, warps, stages, programs per SM


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


assert s.startswith("# okc: problem=muse_attention_decode\n")
first, rest = s.split("\n", 1)
assert rest.startswith('"""')
s = first + "\n" + '''"""v16: v15b with full-attention verify blocks at batch >= 2 (FA3 there, ~1.02x) on a dedicated split-KV kernel,
_attn_verify: per (split, request) program, both KV heads, the 8 tokens x 8 heads of each KV head packed as one 64-row
MMA tile, each key's 128-byte cache row (both heads) read once with V issued alongside K, and the merge plus the
per-head RMSNorm x sigmoid(gate) done in-kernel by the request's last-arriving split (self-resetting counter), so
one launch replaces split + combine. Same precision as v15b's head kernel (FP8 Q.K, BF16 P.V, FP32 softmax / merge).

''' + rest[3:]
rep('''def kernel_fn(data):''', KERNEL + '''def kernel_fn(data):''')
rep('''    if path is None and q.device.type == "cpu":''', f'''    if L != 1 and kind == "full" and batch >= 2 and hkv == 2 and d == 64 and _verify_tile_ok(hq, hkv, L):
        return _verify_full(data, q, qkv, cfg, *{VERIFY_CFG})
    if path is None and q.device.type == "cpu":''')
rep('''def kernel_fn(data):''', '''def _verify_tile_ok(hq, hkv, L):
    """_attn_verify's tile: G * L query rows per KV head, a power of two >= 16."""
    r = (hq // hkv) * L
    return r >= 16 and r & (r - 1) == 0 and (hq * L) & (hq * L - 1) == 0


def kernel_fn(data):''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_attention_decode/runs/v16_verify_fused.py"
dst.write_text(s)
