"""Build muse_decode_layer/runs/v23d_bench.py: decode v23 (runs/v23_fv_attn.py, unchanged kernel_fn) plus a one-time
diagnostic micro-benchmark run at the end of prepare() (untimed; prints only). Purpose: measure on the judge's H200
what limits full-attention decode at batch 16 / 32 (ctx 80000, page 1, agentic lengths), the decode layer's and the
attention problem's 2nd-largest kernel: (a) a pure page-table gather of the K / V rows (no math) = the achievable
gather bandwidth, (b) a contiguous read of the same bytes, (c) the split-KV "rows" kernel over a config sweep
(BLOCK_N, warps, stages, programs per SM, HOIST, PREF), each with cold L2 (a 256 MB buffer read before every rep),
median of 15 reps, CUDA events around the attention kernel + combine only. Synthetic inputs (random E4M3 bytes,
scattered pages); outputs of every rows config are compared with the v19 config's. A time guard stops the sweep at
150 s. Run from the repo root.
"""
import sys

SRC = "muse_decode_layer/runs/v23_fv_attn.py"
DST = "muse_decode_layer/runs/v23d_bench.py"
s = open(SRC).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


BENCH = r'''

# ---- v23d diagnostic (prepare() only; never on the timed path) ----
@triton.jit
def _attn_rows_pref(q_ptr, stride_qt, k_ptr, v_ptr, pt_ptr, stride_pt, seqlens_ptr,
                    ks_ptr, vs_ptr, o_ptr, lse_ptr, qk_scale, chunk, PAGE,
                    L: tl.constexpr, G: tl.constexpr, HKV: tl.constexpr, NSPLIT: tl.constexpr,
                    WINDOW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False,
                    PREF: tl.constexpr = False):
    """attention v23's _attn_split_rows (decode v19's + PREF: next tile's page indices one iteration ahead)."""
    s = tl.program_id(0)
    b = tl.program_id(1)
    HQ: tl.constexpr = HKV * G
    W: tl.constexpr = HKV * D
    m = tl.arange(0, BLOCK_M)
    j = m // HQ
    head = m % HQ
    kvh = head // G
    mvalid = m < L * HQ
    c = tl.arange(0, W)
    seg = c // D
    dd = c % D
    tq = b * L + j
    mine = (seg[None, :] == kvh[:, None]) & mvalid[:, None]
    q = tl.load(q_ptr + tq[:, None] * stride_qt + head[:, None] * D + dd[None, :], mask=mine, other=0.0)
    total = tl.load(seqlens_ptr + b)
    seq = total - L
    pos = seq + j
    lo = s * chunk
    hi = tl.minimum(lo + chunk, total)
    if HW:
        ks = tl.load(ks_ptr)
        vs = tl.load(vs_ptr)
    else:
        ks = tl.load(ks_ptr) * 256.0
        vs = tl.load(vs_ptr) * 256.0
    scale = qk_scale * ks * LOG2E
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, W), tl.float32)
    if PREF:
        t0 = lo + tl.arange(0, BLOCK_N)
        pnext = tl.load(pt_ptr + b * stride_pt + t0 // PAGE, mask=t0 < hi, other=0)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        if PREF:
            page = pnext
            tn = t + BLOCK_N
            pnext = tl.load(pt_ptr + b * stride_pt + tn // PAGE, mask=tn < hi, other=0)
        else:
            page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
        row = (page.to(tl.int64) * PAGE + t % PAGE) * W
        kb = tl.load(k_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        if HOIST:
            vb = tl.load(v_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        sc = tl.dot(q, tl.trans(_kv(kb, HW, q.dtype))) * scale
        ok = tmask[None, :] & (t[None, :] <= pos[:, None])
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(sc - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        if not HOIST:
            vb = tl.load(v_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        if PV16:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), _kv(vb, HW, tl.bfloat16))
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), _kv(vb, HW, q.dtype))
        m_i = m_new
    out = acc * (vs / tl.where(l_i > 0, l_i, 1.0))[:, None]
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.where(l_i > 0, l_i, 1.0)), float("-inf"))
    idx = (tq * HQ + head) * NSPLIT + s
    tl.store(o_ptr + idx[:, None] * D + dd[None, :], out, mask=mine)
    tl.store(lse_ptr + idx, lse, mask=mvalid)


@triton.jit
def _gather_probe(k_ptr, v_ptr, pt_ptr, stride_pt, seqlens_ptr, out_ptr, chunk, NSPLIT: tl.constexpr,
                  BLOCK_N: tl.constexpr, W: tl.constexpr, PREF: tl.constexpr, CONTIG: tl.constexpr):
    """Reads (split s, request b)'s K and V rows through the page table (CONTIG: rows b * stride + t instead) and
    sums the bytes, so the loads can't be dropped. No other math: the gather's own bandwidth."""
    s = tl.program_id(0)
    b = tl.program_id(1)
    c = tl.arange(0, W)
    total = tl.load(seqlens_ptr + b)
    lo = s * chunk
    hi = tl.minimum(lo + chunk, total)
    acc = tl.zeros((BLOCK_N, W), tl.int32)
    if PREF:
        t0 = lo + tl.arange(0, BLOCK_N)
        pnext = tl.load(pt_ptr + b * stride_pt + t0, mask=t0 < hi, other=0)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        if CONTIG:
            page = b * stride_pt + t
        elif PREF:
            page = pnext
            tn = t + BLOCK_N
            pnext = tl.load(pt_ptr + b * stride_pt + tn, mask=tn < hi, other=0)
        else:
            page = tl.load(pt_ptr + b * stride_pt + t, mask=tmask, other=0)
        row = page.to(tl.int64) * W
        kb = tl.load(k_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        vb = tl.load(v_ptr + row[:, None] + c[None, :], mask=tmask[:, None], other=0)
        acc += kb.to(tl.int32) + vb.to(tl.int32)
    tl.store(out_ptr + b * NSPLIT + s, tl.sum(tl.sum(acc, 1), 0))


_DIAG = {}


def _diag_bench():
    """One-time prepare() diagnostic (see the module header). Prints a table; returns nothing."""
    import time
    if _DIAG or not torch.cuda.is_available():
        return
    _DIAG["done"] = True
    t_start = time.time()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1234)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    hq, hkv, d, W = 16, 2, 64, 128

    def timed(fn, reps=15):
        ts = []
        for _ in range(reps):
            flush.view(torch.int32).sum()  # cold L2 by reading > L2, as the judge does (a memset leaves dirty lines)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    print("v23d diag: device", torch.cuda.get_device_name(), "triton", triton.__version__, flush=True)
    for batch in (16, 32):
        ctx = 80000
        u = torch.rand((batch,), generator=g, device=dev)
        lens = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, math.ceil(ctx * 1.075))
        cap = math.ceil(ctx * 1.075) + 1
        total = batch * cap
        perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
        pt = perm.view(batch, cap).contiguous()
        sl = (lens + 1).to(torch.int32)
        kc = (torch.randint(0, 256, (total, W), generator=g, device=dev, dtype=torch.int32) & 0x77).to(torch.uint8)
        vc = (torch.randint(0, 256, (total, W), generator=g, device=dev, dtype=torch.int32) & 0x77).to(torch.uint8)
        q = (2 * torch.randn((batch, hq * d), generator=g, device=dev)).to(FP8)
        gate = torch.randn((batch, hq * d), generator=g, device=dev).to(torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=dev)
        nbytes = 2 * W * int(sl.sum())
        cfg = {"window": 0, "q_len": 1, "batch": batch}
        print(f"batch {batch}: K+V bytes {nbytes / 1e6:.1f} MB (floor at 4.8 TB/s {nbytes / 4.8e6:.1f} us)", flush=True)
        # (a) / (b) gather probes
        for nsp, bn, wp, st, pref, contig in [(128 // (batch // 16), 64, 4, 3, False, False),
                                               (128 // (batch // 16), 128, 4, 3, False, False),
                                               (128 // (batch // 16), 128, 4, 4, True, False),
                                               (128 // (batch // 16), 256, 8, 3, False, False),
                                               (32 // (batch // 16), 128, 4, 4, False, False),
                                               (32 // (batch // 16), 256, 8, 4, True, False),
                                               (128 // (batch // 16), 128, 4, 3, False, True),
                                               (32 // (batch // 16), 256, 8, 4, False, True)]:
            if time.time() - t_start > 150:
                break
            chunk = triton.cdiv(cap, nsp)
            out = torch.empty((batch * nsp,), dtype=torch.int32, device=dev)
            fn = lambda: _gather_probe[(nsp, batch)](kc, vc, pt, pt.stride(0), sl, out, chunk, NSPLIT=nsp, BLOCK_N=bn,
                                                     W=W, PREF=pref, CONTIG=contig, num_warps=wp, num_stages=st)
            fn()
            t = timed(fn)
            print(f"  probe {'contig' if contig else 'gather'} nsplit {nsp} BN {bn} w {wp} st {st} pref {int(pref)}: "
                  f"{t:.1f} us  {nbytes / t / 1e6:.2f} TB/s", flush=True)
        # (c) rows kernel sweep
        ref = None
        for bn, wp, st, occ, hoist, pref in [(64, 4, 3, 16, True, False), (128, 4, 3, 16, True, True),
                                            (64, 4, 3, 16, True, True), (64, 4, 4, 16, True, True),
                                            (128, 4, 4, 16, True, True), (128, 8, 3, 16, True, True),
                                            (64, 4, 3, 8, True, True), (128, 4, 3, 8, True, True),
                                            (128, 4, 4, 8, True, True), (64, 4, 3, 32, True, True),
                                            (128, 4, 3, 4, True, True), (256, 8, 3, 8, True, True),
                                            (64, 4, 2, 16, True, True), (64, 4, 5, 16, True, True),
                                            (64, 2, 3, 16, True, True), (32, 4, 3, 16, True, True),
                                            (32, 4, 4, 32, True, True), (128, 8, 4, 8, True, True)]:
            if time.time() - t_start > 150:
                print("  (time guard: sweep stopped)", flush=True)
                break
            nsp = _nsplit(cfg, batch, 1, cap, occ)
            chunk = triton.cdiv(cap, nsp)
            o_part = torch.empty((batch * hq * nsp, d), device=dev, dtype=torch.float32)
            lse = torch.empty((batch * hq * nsp,), device=dev, dtype=torch.float32)
            out = torch.empty((batch, hq * d), device=dev, dtype=torch.bfloat16)

            def fn():
                _attn_rows_pref[(nsp, batch)](q, q.stride(0), kc, vc, pt, pt.stride(0), sl, one, one, o_part, lse,
                                              0.125, chunk, 1, L=1, G=8, HKV=2, NSPLIT=nsp, WINDOW=0, BLOCK_M=16,
                                              BLOCK_N=bn, D=d, HW=True, HOIST=hoist, PV16=True, PREF=pref,
                                              num_warps=wp, num_stages=st)
                _combine_norm_gate[(batch, hq)](o_part, lse, gate, gate.stride(0), out, RMS_EPS, HQ=hq, NSPLIT=nsp,
                                                D=d, num_warps=4 if nsp >= 64 else 1)
            fn()
            torch.cuda.synchronize()
            if ref is None:
                ref = out.float().clone()
            err = ((out.float() - ref).norm() / ref.norm()).item()
            t = timed(fn)
            print(f"  rows BN {bn} w {wp} st {st} occ {occ} (nsplit {nsp}) hoist {int(hoist)} pref {int(pref)}: "
                  f"{t:.1f} us  {nbytes / t / 1e6:.2f} TB/s  rel.err vs first {err:.1e}", flush=True)
        del kc, vc, pt, perm
        torch.cuda.empty_cache()
    del flush
    torch.cuda.empty_cache()
    print(f"v23d diag: done in {time.time() - t_start:.0f} s", flush=True)

'''
s = rep(s, "\ndef prepare(weights):", BENCH + "\ndef prepare(weights):")
# call at the end of prepare (its only top-level `    return state`)
i = s.index("\ndef prepare(weights):")
j = s.index("\n    return state\n", i)
s = s[:j] + "\n    try:\n        _diag_bench()  # v23d: diagnostic only, untimed\n    except Exception as e:  # never fail the run\n        print(\"v23d diag failed:\", repr(e)[:500], flush=True)" + s[j:]
if "\nimport math\n" not in s:
    s = rep(s, "\nimport types\n", "\nimport math\nimport types\n")
s = rep(s, '# okc: problem=muse_decode_layer\n"""v23:', '''# okc: problem=muse_decode_layer
"""v23d (DIAGNOSTIC, not for the board): v23 plus a one-time untimed micro-benchmark at the end of prepare() that
prints gather-bandwidth probes and a split-KV rows-kernel config sweep for full decode at batch 16 / 32
(tools/build_decode_v23d.py). kernel_fn is v23's, unchanged.

v23:''')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
