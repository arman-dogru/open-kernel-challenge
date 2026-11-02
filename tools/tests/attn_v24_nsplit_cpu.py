"""CPU (TRITON_INTERPRET) check for attention v24: full decode at batch 17 / 33 where v24's ~4 programs/SM gives a
different split count than v23's (32 vs 64, 16 vs 64): both outputs vs the reference (rel. L2) and v24 vs v23.
(The full check_implementation trips on new cache rows here: the Triton CPU interpreter's E4M3 cast, STATUS r6 n2.)"""
import importlib.util, os, sys
os.environ["TRITON_INTERPRET"] = "1"
pdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "muse_attention_decode"); sys.path.insert(0, pdir)
import torch, reference as R
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(pdir, p)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A, B = load("runs/v23_fv_addr.py", "a"), load("runs/v24_rows_occ4.py", "b")
for c in [dict(batch=17, ctx=12000, heads="tp8", layer="full", spread="agentic"), dict(batch=33, ctx=8000, heads="tp8", layer="full", spread="mixed", page=1)]:
    outs = []
    for M in (A, B):
        data = R.generate_input(seed=5, device="cpu", **c)
        outs.append(M.kernel_fn(data).float())
    ref = R.ref_kernel(R.generate_input(seed=5, device="cpu", **c))[0].float().reshape(outs[0].shape)
    e = lambda o: ((o - ref).norm() / ref.norm()).item()
    print(c, "v23 vs ref %.4f  v24 vs ref %.4f  |v24 - v23| rel %.2e" % (e(outs[0]), e(outs[1]), ((outs[1]-outs[0]).norm()/outs[0].norm()).item()), flush=True)
