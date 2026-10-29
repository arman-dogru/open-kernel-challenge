"""Builds muse_attention_decode/runs/v18_head_merge.py from v17_rows_prefetch.py (exact replacements, each asserted
once). Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = [(ROOT / "muse_attention_decode/runs/v17_rows_prefetch.py").read_text()]


def rep(a, b):
    assert S[0].count(a) == 1, (a[:70], S[0].count(a))
    S[0] = S[0].replace(a, b)


rep('''"""v17: v15b''', '''"""v18: v17 with the split-KV "head" kernel (sliding decode / verify at any batch, full verify at batch 1) merging its
own splits: after writing its partial, a program arrives on a self-resetting counter per (request, KV head), and the
last arrival merges that head group's NSPLIT partials and applies the per-head RMSNorm x sigmoid(gate), exactly as
_combine_norm_gate does (same FP32 math). That removes the combine launch from those cells, which are latency-bound
(12-27 us). Same merge scheme as attention v16's _attn_verify (check 9f59e5963dd740bd, 16/16 tests).

v17: v15b''')
rep('''                D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False):''',
    '''                D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False,
                cnt_ptr=None, gate_ptr=None, stride_gt=0, out_ptr=None, eps=0.0, MERGE: tl.constexpr = False):''')
rep('''    tl.store(o_ptr + idx[:, None] * D + d[None, :], out, mask=mvalid[:, None])
    tl.store(lse_ptr + idx, lse, mask=mvalid)
''', '''    tl.store(o_ptr + idx[:, None] * D + d[None, :], out, mask=mvalid[:, None])
    tl.store(lse_ptr + idx, lse, mask=mvalid)
    if MERGE:  # last arrival of (request b, KV head h) merges its NSPLIT partials (v18), as _combine_norm_gate
        tl.debug_barrier()  # every thread's partial stores issued before the one acq_rel GPU-scope arrival
        old = tl.atomic_add(cnt_ptr + b * HKV + h, 1, sem="acq_rel", scope="gpu")
        if old == NSPLIT - 1:
            sp = tl.arange(0, NSPLIT)
            base = (tq * HQ + head) * NSPLIT
            lse_t = tl.load(lse_ptr + base[:, None] + sp[None, :], mask=mvalid[:, None], other=float("-inf"),
                            cache_modifier=".cg")
            mx = tl.max(lse_t, 1)
            mxs = tl.where(mx == float("-inf"), 0.0, mx)
            wsum = tl.sum(tl.where(lse_t == float("-inf"), 0.0, tl.exp2(lse_t - mxs[:, None])), 1)
            acc2 = tl.zeros((BLOCK_M, D), tl.float32)
            for k in range(NSPLIT):
                lk = tl.load(lse_ptr + base + k, mask=mvalid, other=float("-inf"), cache_modifier=".cg")
                wk = tl.where(lk == float("-inf"), 0.0, tl.exp2(lk - mxs))
                ok_ = tl.load(o_ptr + (base + k)[:, None] * D + d[None, :], mask=mvalid[:, None], other=0.0,
                              cache_modifier=".cg")
                acc2 += ok_ * wk[:, None]
            a = acc2 / tl.where(wsum > 0, wsum, 1.0)[:, None]
            rr = a * tl.rsqrt(tl.sum(a * a, 1) / D + eps)[:, None]
            g = tl.load(gate_ptr + tq[:, None] * stride_gt + head[:, None] * D + d[None, :], mask=mvalid[:, None],
                        other=0.0).to(tl.float32)
            tl.store(out_ptr + tq[:, None] * (HQ * D) + head[:, None] * D + d[None, :],
                     (rr * tl.sigmoid(g)).to(out_ptr.dtype.element_ty), mask=mvalid[:, None])
            tl.store(cnt_ptr + b * HKV + h, 0)  # self-reset for the next call / graph replay
''')
rep('''    else:
        _attn_split[(hkv, nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"], BLOCK_M=block_m, BLOCK_N=block_n, D=d,
            HW=hw, HOIST=hoist, PV16=hw, num_warps=warps, num_stages=stages)
''', '''    else:  # head kernel with the in-kernel merge + norm / gate (v18): no combine launch
        gate = qkv[:, (hq + 2 * hkv) * d:]
        out = torch.empty((T, hq * d), device=qkv.device, dtype=data["qkv"].dtype)
        _attn_split[(hkv, nsplit, batch)](
            *args, L=L, G=G, HKV=hkv, NSPLIT=nsplit, WINDOW=cfg["window"], BLOCK_M=block_m, BLOCK_N=block_n, D=d,
            HW=hw, HOIST=hoist, PV16=hw, cnt_ptr=_merge_counters(qkv.device, batch * hkv), gate_ptr=gate,
            stride_gt=qkv.stride(0), out_ptr=out, eps=RMS_EPS, MERGE=True, num_warps=warps, num_stages=stages)
        return out.view(batch, L, hq * d)
''')
rep('''def kernel_fn(data):''', '''_MERGE_CNT = {}


def _merge_counters(device, n):
    """Arrival counters for the head kernel's in-kernel merge, zero between calls (each last arrival resets its own).
    Not derived from the inputs."""
    cnt = _MERGE_CNT.get(device)
    if cnt is None or cnt.numel() < n:
        cnt = _MERGE_CNT[device] = torch.zeros((max(n, 512),), device=device, dtype=torch.int32)
    return cnt


def kernel_fn(data):''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_attention_decode/runs/v18_head_merge.py"
dst.write_text(S[0])
