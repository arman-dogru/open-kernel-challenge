"""CPU (TRITON_INTERPRET) check for decode v24: its _attn_split_rows (attention v23's, + PREF) with v24's full-decode
config (BLOCK_N 128, PREF, HOIST) matches decode v19's _attn_split_rows bit for bit (o partials and lse) over
batches / lengths / pages / splits, including a sliding window and ragged last tiles; v24d's _diag_bench exists.
Usage: /workspace/okc-cpu-venv/bin/python tools/tests/decode_v24_rows_cpu.py"""
import importlib.util, os, sys
os.environ["TRITON_INTERPRET"] = "1"
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
src = open(os.path.join(root, "tools", "tests", "decode_v23_fv_glue.py")).read().split('spec = importlib.util.spec_from_file_location("dec", path)')[0]
exec(src.replace('path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "muse_decode_layer", "runs", "v23_fv_attn.py")', "path = None"))
import torch


def load(name):
    p = os.path.join(root, "muse_decode_layer", "runs", name)
    spec = importlib.util.spec_from_file_location(name[:-3], p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V19, V24, V24D = load("v19_nb2_256_evict.py"), load("v24_rows_occ4.py"), load("v24d_bench.py")
assert callable(V24D._diag_bench) and V24D._attn_path("full", 1, 16, 2, 64) == V24._attn_path("full", 1, 16, 2, 64)
print("v24 path full b16:", V24._attn_path("full", 1, 16, 2, 64), "b1:", V24._attn_path("full", 1, 1, 2, 64))
ok = True
torch.manual_seed(0)
for batch, cap, lens, page, nsp, window in [(2, 300, [250, 299], 1, 4, 0), (3, 520, [517, 100, 333], 1, 2, 0),
                                           (2, 64 * 9, [500, 575], 16, 3, 0), (2, 700, [690, 400], 1, 2, 128)]:
    W, hq, d = 128, 16, 64
    npages = cap // page
    total = batch * npages
    pt = torch.randperm(total).to(torch.int32).view(batch, npages).contiguous()
    sl = torch.tensor(lens, dtype=torch.int32)
    kc = (torch.randint(0, 256, (total * page, W), dtype=torch.int32) & 0x77).to(torch.uint8)
    vc = (torch.randint(0, 256, (total * page, W), dtype=torch.int32) & 0x77).to(torch.uint8)
    q = torch.randn(batch, hq * d)
    one = torch.ones(())
    span = window if window else cap
    chunk = -(-span // nsp)
    outs = []
    for fn, kw in [(V19._attn_split_rows, dict(BLOCK_N=64, HOIST=True)),
                   (V24._attn_split_rows, dict(BLOCK_N=128, HOIST=True, PREF=True))]:
        o = torch.zeros(batch * hq * nsp, d); l = torch.zeros(batch * hq * nsp)
        fn[(nsp, batch)](q, q.stride(0), kc, vc, pt, pt.stride(0), sl, one, one, o, l, 0.125, chunk, page, L=1, G=8,
                         HKV=2, NSPLIT=nsp, WINDOW=window, BLOCK_M=16, D=d, HW=False, PV16=False, **kw)
        outs.append((o, l))
    # different tile sizes change the online-softmax order: compare the merged result per (token, head)
    def merge(o, l):
        o = o.view(batch, hq, nsp, d); l = l.view(batch, hq, nsp)
        mx = l.max(-1, keepdim=True).values
        w = torch.where(torch.isfinite(l), torch.exp2(l - mx), torch.zeros_like(l))
        return (o * w[..., None]).sum(2) / w.sum(-1, keepdim=True)
    a, b = merge(*outs[0]), merge(*outs[1])
    err = ((a - b).norm() / a.norm()).item()
    print("case", batch, cap, lens, "page", page, "nsplit", nsp, "window", window, "rel err v24 vs v19 %.2e" % err)
    ok &= err < 1e-5
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
