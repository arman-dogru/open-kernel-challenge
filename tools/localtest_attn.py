# Local logic test on CPU (TRITON_INTERPRET=1): small contexts, fp32 queries (no bf16 in the
# interpreter), a torch stand-in for SGLang's norm/RoPE/cache-store. Checks indexing, masks,
# splits and the combine, not H200 speed.
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "muse_attention_decode"); sys.path.insert(0, d)
import torch, reference as R, submission as S
cases = [dict(batch=3, ctx=3000, heads="tp8", layer="full"), dict(batch=17, ctx=1800, heads="tp8", layer="full", spread="agentic"), dict(batch=5, ctx=3000, heads="tp8", layer="sliding", page=1), dict(batch=1, ctx=2900, heads="tp8", layer="full", page=1),dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
         dict(batch=2, ctx=2047, heads="tp8", layer="sliding"),
         dict(batch=2, ctx=700, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
         dict(batch=2, ctx=3000, heads="tp8", layer="full"),
         dict(batch=2, ctx=600, heads="dp", layer="full", q_len=8),
         dict(batch=2, ctx=2500, heads="tp8", layer="sliding", dist="edges", page=1),
         dict(batch=2, ctx=900, heads="tp8", layer="full", q_len=8, page=1, kv_scale=0.015625)]
for c in cases:
    data = R.generate_input(seed=11, device="cpu", **c)
    out = S.kernel_fn(data)
    raw = R.collect(data, out.to(torch.bfloat16))
    print(c, R.check_implementation(data, raw), flush=True)
