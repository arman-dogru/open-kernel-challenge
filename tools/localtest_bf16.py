# Local logic test on CPU (TRITON_INTERPRET=1). The interpreter can't do bf16, so x and the
# weights run as fp32 copies; this checks indexing/masking/split-K logic, not bf16 precision.
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
d = "/workspace/open-kernel-challenge/muse_bf16_proj"; sys.path.insert(0, d)
import torch, reference as R, submission as S
W = R.generate_weights(1, device="cpu")
st = S.prepare(W)
st32 = {k: (v.float() if v.dtype == torch.bfloat16 else v) for k, v in st.items()}
for m, layer, dist in [(1,"qkv_gate","normal"),(7,"pooler","normal"),(33,"o","normal"),(5,"unpooler","normal"),(130,"unpooler","normal"),(8,"router","normal"),(9,"router","outliers"),(64,"router","normal")]:
    data = R.generate_input(m=m, layer=layer, seed=3, dist=dist, device="cpu")
    d32 = dict(data); d32["x"] = data["x"].float()
    out = S.kernel_fn(d32, st32)
    if layer != "router": out = out.to(torch.bfloat16)
    print(m, layer, dist, tuple(out.shape), R.check_implementation(data, out, W))
