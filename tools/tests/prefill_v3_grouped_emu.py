"""Test of prefill v3's >= 8192-token grouped expert path (review of #16). Runs the path's own kernels (from
tools/prefill_v3_experts.py.txt, the text built into runs/v3_gs_after_local.py): NVFP4 dequant, local-route gather,
SiLU x tensor scale, un-sort x tensor scale, with torch._grouped_mm, against an FP32 reference per route built
from nvfp4.dequantize. On CPU (TRITON_INTERPRET=1, small shapes) torch._grouped_mm is emulated by a loop over the
per-expert row offsets; on an sm_90 GPU the real op runs. Also checks that the dequantized weights are exact (no
BF16 rounding before the GEMM) and that non-local slots are never written.
    TRITON_INTERPRET=1 python tools/tests/prefill_v3_grouped_emu.py      (CPU)
    python tools/tests/prefill_v3_grouped_emu.py --cuda                  (GPU)"""
import os
import sys
from pathlib import Path

CUDA = "--cuda" in sys.argv
if not CUDA:
    os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "muse_prefill_layer"))
import nvfp4  # noqa: E402

dev = "cuda" if CUDA else "cpu"
if not CUDA and not hasattr(torch, "_grouped_mm"):
    def _grouped_mm_emu(a, b, offs=None):
        out = torch.full((a.shape[0], b.shape[2]), float("nan"), dtype=a.dtype)
        start = 0
        for e, end in enumerate(offs.tolist()):
            out[start:end] = (a[start:end].float() @ b[e].float()).to(a.dtype)
            start = end
        return out
    torch._grouped_mm = _grouped_mm_emu

TOP_K, E, H, I = 8, 32, 64, 64  # small hidden / expert sizes (powers of two, as the kernels' aranges need)
ns = dict(torch=torch, triton=triton, tl=tl, TOP_K=TOP_K, LOCAL_EXPERTS=E, POST_NORM_EPS=1e-8)
exec((ROOT / "tools/prefill_v3_experts.py.txt").read_text(), ns)

g = torch.Generator().manual_seed(0)
w13q, w13sf, _ = nvfp4.random_weights(2 * I, H, generator=g, device="cpu", leading=(E,))
w2q, w2sf, _ = nvfp4.random_weights(H, I, generator=g, device="cpu", leading=(E,))
gs13 = torch.rand(E, generator=g) * 1e-3 + 1e-4
gs2 = torch.rand(E, generator=g) * 1e-3 + 1e-4

# 1. dequant without the tensor scale is exact
deq = ns["_dequant"](w13q.to(dev), w13sf.to(dev)).cpu()
exact = torch.cat([nvfp4.dequantize(w13q[e], w13sf[e], 1.0)[None] for e in range(E)])
assert torch.equal(deq.float(), exact), "dequant (no tensor scale) is not exact in BF16"
print("dequant exact: ok")

# 2. the path up to the un-sorted expert outputs, 300 tokens with ~1/8 of the routes local (as on one EP rank)
T = 300
x = torch.randn(T, H, generator=g).to(torch.bfloat16)
ids = torch.stack([torch.randperm(256, generator=g)[:TOP_K] for _ in range(T)])
local = torch.where(ids < E, ids, -1).to(torch.int32)
flat = local.view(-1)
key = torch.where(flat >= 0, flat, E)
skey, perm = torch.sort(key, stable=True)
counts = torch.zeros(E + 1, dtype=torch.int32).scatter_add_(0, key, torch.ones_like(key, dtype=torch.int32))
offs = torch.cumsum(counts[:E], 0, dtype=torch.int32)
rows = flat.numel()
d = lambda t: t.to(dev)  # noqa: E731
xs = torch.zeros((rows, H), dtype=torch.bfloat16, device=dev)
ns["_gather_local_kernel"][(rows,)](d(x), d(perm), d(offs), xs, TOPK=TOP_K, NE=E, H=H)
n_local = int(offs[-1])
assert torch.equal(xs[:n_local].cpu(), x[perm[:n_local] // TOP_K]) and not xs[n_local:].any(), "gather"
w13 = ns["_dequant"](d(w13q), d(w13sf))
h1 = torch._grouped_mm(xs, w13.transpose(1, 2), offs=d(offs))
eid = torch.where(skey < E, skey, -1).to(torch.int32)
h2 = torch.zeros((rows, I), dtype=torch.bfloat16, device=dev)
ns["_silu_mul_gs_kernel"][(rows,)](h1, h2, d(eid), d(gs13), I=I)
w2 = ns["_dequant"](d(w2q), d(w2sf))
ys = torch._grouped_mm(h2, w2.transpose(1, 2), offs=d(offs))
inv = torch.empty_like(perm)
inv[perm] = torch.arange(rows)
sentinel = 1024.0  # exact in BF16
y = torch.full((rows, H), sentinel, dtype=torch.bfloat16, device=dev)
ns["_unsort_gs_kernel"][(rows,)](ys, d(inv), d(flat), d(gs2), y, H=H)
y = y.cpu().float()

# FP32 reference per route with the true (scaled) weights
ref = torch.zeros(rows, H)
W13 = exact * gs13[:, None, None]
W2 = torch.cat([nvfp4.dequantize(w2q[e], w2sf[e], 1.0)[None] for e in range(E)]) * gs2[:, None, None]
for j in range(rows):
    e = int(flat[j])
    if e < 0:
        continue
    xr = x[j // TOP_K].float()
    gu = W13[e] @ xr
    hr = torch.nn.functional.silu(gu[:I]) * gu[I:]
    ref[j] = W2[e] @ hr
m = flat >= 0
assert (y[~m] == sentinel).all(), "non-local slots must not be written"
err = ((y[m] - ref[m]).norm() / ref[m].norm()).item()
print(f"local routes {n_local} of {rows}; relative L2 error vs FP32 reference {err:.2e}")
assert err < 1e-2, err
print("ok")
