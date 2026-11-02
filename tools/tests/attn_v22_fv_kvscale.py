"""Review #20 m2: the v22 full-verify path (q_len 8, batch >= 2, tp8) with kv_scale != 1, and K / V scales that differ.

The judge's tests only use kv_scale 0.015625 on sliding verify and full decode, never on full verify, so the CUDA
kernel's two scale uses (k_scale folded into the exp2 scale, v_scale into o = acc * v_scale / l) are covered here.

Part A (CPU, no GPU; run with okc-cpu-venv): a step-by-step emulation of tools/attn_v22_fv.cu per (split, request):
64-key tiles, position mask, exp2 online softmax with the +8 bias, P rounded to E4M3 (satfinite), raw E4M3 V,
row sums of the rounded P (V^T ones rows), o = acc * v_scale / l, lse = m + log2(l) - 8. Its partials go through the
submission's own Triton _combine_norm_gate (TRITON_INTERPRET=1) and are checked with reference.check_implementation.
It checks the kernel's math and scale handling, not its CUDA layout (swizzles, fragment order). New cache rows come
from the reference here (the interpreter's E4M3 cast is wrong, STATUS.md r6 n2).
Part B (an sm_90 GPU only; skipped elsewhere): runs submission.kernel_fn on the same cases, asserts the CUDA _fv path
was taken, checks the output, then captures kernel_fn in a CUDA graph and replays it (no compile inside capture, m3).
Note: a uniform v_scale error cancels in the output (the combine's per-head RMSNorm), so the check catches k_scale
misuse (verified: dropping k_scale fails all 4 cases at 0.31-0.52 rel. L2) but not a v_scale factor alone.
Usage: /workspace/okc-cpu-venv/bin/python tools/tests/attn_v22_fv_kvscale.py [runs/file.py]      (part A, CPU)
       ATTN_GPU=1 python tools/tests/attn_v22_fv_kvscale.py [runs/file.py]                        (part B, sm_90)"""
import importlib.util, math, os, sys
if not os.environ.get("ATTN_GPU"):  # part A runs the Triton combine in the CPU interpreter
    os.environ.setdefault("TRITON_INTERPRET", "1")
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
pdir = os.path.join(root, "muse_attention_decode")
sys.path.insert(0, pdir)
import torch
import reference as R
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(pdir, "runs", "v22_cuda_fv.py")
spec = importlib.util.spec_from_file_location("submission", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# (generate_input kwargs, v_scale override or None). kv_scale sets both scales; the override makes V's differ.
CASES = [(dict(batch=2, ctx=900, heads="tp8", layer="full", q_len=8, kv_scale=0.015625), None),
         (dict(batch=3, ctx=1500, heads="tp8", layer="full", q_len=8, kv_scale=0.37, page=1), None),
         (dict(batch=2, ctx=700, heads="tp8", layer="full", q_len=8, kv_scale=0.015625, spread="mixed"), 0.05),
         (dict(batch=4, ctx=600, heads="tp8", layer="full", q_len=8, kv_scale=2.5, dist="edges", page=16), 0.6)]
E4 = torch.float8_e4m3fn
BN = 64


def _make(kw, vs, device):
    data = R.generate_input(seed=7, device=device, **kw)
    if vs is not None:  # both kernel and reference read v_scale from data (cache values reinterpreted)
        data["v_scale"] = torch.tensor(vs, dtype=torch.float32, device=device)
    assert data["config"]["q_len"] == 8 and (data["k_scale"].item() != 1.0)
    return data


def emulate(data):
    """Partials of attn_fv (see tools/attn_v22_fv.cu) for every (split, request), then the Triton combine."""
    cfg = data["config"]
    batch, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    assert (hq, hkv, d, L) == (16, 2, 64, 8)
    G, T = hq // hkv, batch * L
    qkv = data["qkv"].view(T, -1)
    q = S._norm_rope_store_fused(data, qkv.float(), cfg, torch.float32)  # q (fp32) + the new K/V, like kernel_fn
    # The Triton 3.9 CPU interpreter's fp32 -> E4M3 cast is wrong on exponent carries (STATUS.md, r6 n2), so the
    # new cache rows it stores fail the 8% cache check on CPU; store the reference's own E4M3 rows instead (the
    # kernel under test only reads the cache). On the GPU (part B) the real fused store is used.
    _, rk, rv = R.ref_kernel(data)
    slots = R.new_slots(data)
    data["k_cache"].view(-1, hkv, d)[slots] = rk
    data["v_cache"].view(-1, hkv, d)[slots] = rv
    q = q.view(T, -1)[:, : hq * d].to(E4).float().view(batch, L, hq, d)  # FP8 queries, as on the GPU
    page = cfg["page_size"]
    capacity = data["page_table"].shape[1] * page
    nsplit = S._nsplit(cfg, batch, 1, capacity, S.FV_OCC)
    chunk = -(-capacity // nsplit)
    kc, vc = data["k_cache"].float(), data["v_cache"].float()  # raw E4M3 values, [slots, hkv, d]
    kc, vc = kc.view(-1, hkv, d), vc.view(-1, hkv, d)
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * data["k_scale"].item() * (1 / math.log(2))
    vsc = data["v_scale"].item()
    o_part = torch.zeros((T * hq * nsplit, d))
    lse = torch.full((T * hq * nsplit,), float("-inf"))
    pt, sl = data["page_table"].long(), data["cache_seqlens"].long()
    for b in range(batch):
        total = int(sl[b]); seq = total - L
        pos = seq + torch.arange(L)  # per token
        for s in range(nsplit):
            lo, hi = s * chunk, min(s * chunk + chunk, total)
            for h in range(hkv):
                qh = q[b, :, h * G:(h + 1) * G].reshape(L * G, d)  # rows: token * 8 + head (order irrelevant here)
                rpos = pos.repeat_interleave(G)
                m = torch.full((L * G,), float("-inf")); acc = torch.zeros((L * G, d)); l = torch.zeros(L * G)
                for base in range(lo, hi, BN):
                    t = torch.arange(base, base + BN)
                    valid = t < hi
                    tc = torch.where(valid, t, torch.zeros_like(t))
                    slot = pt[b, tc // page] * page + tc % page
                    k = torch.where(valid[:, None], kc[slot, h], 0.0)  # cp.async zero-fills keys past hi
                    v = torch.where(valid[:, None], vc[slot, h], 0.0)
                    sc = qh @ k.T
                    sc = torch.where((t[None, :] < hi) & (t[None, :] <= rpos[:, None]), sc, float("-inf"))
                    mn = torch.maximum(m, sc.max(1).values * scale)
                    ms = torch.where(mn == float("-inf"), 0.0, mn)
                    a = torch.exp2(m - ms)
                    p = torch.exp2(sc * scale + (8.0 - ms)[:, None])
                    p8 = p.clamp(max=448.0).to(E4).float()  # cvt.rn.satfinite.e4m3
                    acc = acc * a[:, None] + p8 @ v
                    l = l * a + p8.sum(1)
                    m = mn
                f = vsc / torch.where(l > 0, l, 1.0)
                tok = torch.arange(L).repeat_interleave(G); head = h * G + torch.arange(G).repeat(L)
                idx = ((b * L + tok) * hq + head) * nsplit + s
                o_part[idx] = acc * f[:, None]
                lse[idx] = torch.where(l > 0, m + torch.log2(torch.where(l > 0, l, 1.0)) - 8.0, float("-inf"))
    gate = qkv[:, (hq + 2 * hkv) * d:].float().contiguous()  # the interpreter has no bf16
    out = torch.empty((T, hq * d), dtype=torch.float32)
    S._combine_norm_gate[(T, hq)](o_part, lse, gate, gate.stride(0), out, S.RMS_EPS, HQ=hq, NSPLIT=nsplit, D=d,
                                  num_warps=1)
    return out.view(batch, L, hq * d)


def part_a():
    ok = True
    for kw, vs in CASES:
        data = _make(kw, vs, "cpu")
        res = R.check_implementation(data, R.collect(data, emulate(data).to(torch.bfloat16)))
        print("A", kw, "v_scale", vs if vs is not None else kw["kv_scale"], res, flush=True)
        ok &= bool(res[0])
    return ok


def part_b():
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)):
        print("B skipped: needs an sm_90 GPU (H100/H200)")
        return True
    calls = []
    fv = S._fv
    S._fv = lambda *a: (calls.append(1), fv(*a))[1]
    ok = True
    for kw, vs in CASES:
        data = _make(kw, vs, "cuda")
        n0 = len(calls)
        out = S.kernel_fn(data)
        assert len(calls) == n0 + 1, "full-verify CUDA path not taken"
        res = R.check_implementation(data, R.collect(data, out))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = S.kernel_fn(data)
        g.replay(); torch.cuda.synchronize()
        print("B", kw, "v_scale", vs if vs is not None else kw["kv_scale"], res, "graph replay finite",
              bool(torch.isfinite(gout.float()).all()), flush=True)
        ok &= bool(res[0])
    return ok


if __name__ == "__main__":
    ok = part_b() if os.environ.get("ATTN_GPU") else part_a()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
