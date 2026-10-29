# Local logic test on CPU (TRITON_INTERPRET=1) with shrunken dims (fp32 activations; the
# interpreter can't do bf16). Checks routing/indexing/dequant, not H200 precision or speed.
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "muse_moe_w4a16"); sys.path.insert(0, d); sys.path.insert(0, d + "/runs")
import importlib, torch, reference as R
S = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "submission")
R.D_HIDDEN = 256; R.D_EXPERT = 256
W = R.generate_weights(5, device="cpu")
st = S.prepare(W)
for bs, dist in [(1, "normal"), (4, "normal"), (33, "normal"), (20, "skewed")]:
    data = R.generate_input(bs=bs, sharding="ep8", seed=7, dist=dist, device="cpu")
    data["config"]["d_hidden"] = 256; data["config"]["d_expert"] = 256
    d32 = dict(data); d32["hidden_states"] = data["hidden_states"].float()
    out = S.kernel_fn(d32, st)
    print(bs, dist, tuple(out.shape), R.check_implementation(data, out.to(torch.bfloat16), W))
