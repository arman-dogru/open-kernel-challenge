"""CPU (TRITON_INTERPRET=1) output comparison of two attention runs files on the localtest_attn cases: max |a - b| of
kernel_fn's outputs and of the new cache rows. Replaces the /tmp/attn_cmp19.py cited in PR #20 (review nit).
Usage: /workspace/okc-cpu-venv/bin/python tools/tests/attn_cpu_cmp.py runs/a.py runs/b.py   (paths under
muse_attention_decode/ or absolute)"""
import importlib.util, os, sys
os.environ["TRITON_INTERPRET"] = "1"
pdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "muse_attention_decode")
sys.path.insert(0, pdir)
import torch
import reference as R

CASES = [dict(batch=3, ctx=3000, heads="tp8", layer="full"),
         dict(batch=17, ctx=1800, heads="tp8", layer="full", spread="agentic"),
         dict(batch=5, ctx=3000, heads="tp8", layer="sliding", page=1),
         dict(batch=1, ctx=2900, heads="tp8", layer="full", page=1),
         dict(batch=2, ctx=700, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
         dict(batch=2, ctx=3000, heads="tp8", layer="full"),
         dict(batch=2, ctx=600, heads="dp", layer="full", q_len=8),
         dict(batch=2, ctx=900, heads="tp8", layer="full", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=3, ctx=1200, heads="tp8", layer="full", q_len=8)]


def load(path, name):
    p = path if os.path.isabs(path) else os.path.join(pdir, path)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A, B = load(sys.argv[1], "sub_a"), load(sys.argv[2], "sub_b")
worst = 0.0
for c in CASES:
    outs = []
    for M in (A, B):
        data = R.generate_input(seed=11, device="cpu", **c)
        outs.append(R.collect(data, M.kernel_fn(data).to(torch.bfloat16)))
    same = torch.equal(outs[0], outs[1])
    n = c["batch"] * c.get("q_len", 1) * 16 * 64 * 2  # bytes of the bf16 output part (tp8)
    if c["heads"] == "dp":
        n = c["batch"] * c.get("q_len", 1) * R.HEADS["dp"][0] * 64 * 2
    da = outs[0][:n].view(torch.bfloat16).float() - outs[1][:n].view(torch.bfloat16).float()
    md = da.abs().max().item()
    worst = max(worst, md)
    print(c, "bytes identical" if same else "DIFFER", "out maxdiff", md, flush=True)
print("maxdiff", worst)
sys.exit(0 if worst == 0 else 1)
