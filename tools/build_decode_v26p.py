"""Build muse_decode_layer/runs/v26p_bench.py (DIAGNOSTIC): decode v25 (runs/v25_ws4_fv.py, kernel_fn unchanged) +
a one-time, untimed sweep at the end of prepare() of the dense projections at T 1 / 8 / 16 / 32 on the judge's
weights: what v25 runs now (cuBLAS via _gemm for qkv_gate / unpooler, _swapab_gemm PROJ_CFG for o / pooler) vs
_swapab_gemm configs (BN, BK, SPLIT_K, warps, stages); split-K includes its zero fill and BF16 cast. Cold L2 (256 MB
read between reps), median of 15. Prints only. Run from the repo root."""
SRC, DST = "muse_decode_layer/runs/v25_ws4_fv.py", "muse_decode_layer/runs/v26p_bench.py"
s = open(SRC).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


BENCH = r'''

# ---- v26p diagnostic (prepare() only; never on the timed path) ----
_DIAGP = {}


def _diag_proj(dense):
    import time
    if _DIAGP or not torch.cuda.is_available():
        return
    _DIAGP["done"] = True
    t0 = time.time()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(4321)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)

    def timed(f, reps=15):
        ts = []
        for _ in range(reps):
            flush.view(torch.int32).sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            f()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    def sw(x, w, bn, bk, split, warps, stages):
        m, k = x.shape
        n = w.shape[0]
        bm = 16 if m <= 16 else 32
        if split > 1:
            out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
        else:
            out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        _swapab_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm), split)](
            x, w, w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
            BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_LO=False, OUT_BF16=split == 1,
            num_warps=warps, num_stages=stages)
        return out.to(x.dtype) if split > 1 else out

    print("v26p diag:", torch.cuda.get_device_name(), "triton", triton.__version__, flush=True)
    confs = []
    for bn in (32, 64, 128):
        for bk in (128, 256):
            for split in (1, 2, 4, 8):
                for warps in (4,):
                    for stages in (3, 4):
                        confs.append((bn, bk, split, warps, stages))
    layers = [("qkv_gate", dense["qkv_gate"], None), ("o", dense["o"], "o"), ("pooler", dense["pooler"], "pooler"),
              ("unpooler", dense["unpooler"], None), ("unpooler_s", dense["unpooler"][:UNPOOL_SLICE].contiguous(), None)]
    for name, w, cur in layers:
        for m in ((1, 8) if name == "unpooler_s" else (1, 8, 16, 32) if name != "unpooler" else (16, 32)):
            if time.time() - t0 > 230:
                print("  (time guard)", flush=True)
                break
            x = (torch.randn((m, w.shape[1]), generator=g, device=dev) * 0.5).to(torch.bfloat16)
            ref = (x.float() @ w.float().T)
            if cur is None:
                f0 = lambda: _gemm(x, w)
                lab = "cuBLAS _gemm"
            else:
                f0 = lambda: _proj(x, w, cur)
                lab = "PROJ_CFG %s %s" % (cur, PROJ_CFG[cur])
            t_cur = timed(f0)
            best = (t_cur, lab)
            res = []
            for c in confs:
                if c[2] > 1 and (w.shape[1] // c[2]) % c[1]:
                    continue
                try:
                    o = sw(x, w, *c)
                    torch.cuda.synchronize()
                except Exception as e:
                    continue
                err = ((o.float() - ref).norm() / ref.norm()).item()
                if err > 1e-2:
                    print(f"  {name} m {m} {c}: err {err:.1e} (skipped)", flush=True)
                    continue
                t = timed(lambda: sw(x, w, *c))
                res.append((t, c))
                if t < best[0]:
                    best = (t, c)
            res.sort()
            print(f"v26p {name} m {m}: current {lab} {t_cur:.2f} us; best {best[1]} {best[0]:.2f} us; top5 "
                  + " ".join(f"{c}:{t:.2f}" for t, c in res[:5]), flush=True)
    del flush
    torch.cuda.empty_cache()
    print(f"v26p diag: done in {time.time() - t0:.0f} s", flush=True)

'''
s = rep(s, "\ndef prepare(weights):", BENCH + "\ndef prepare(weights):")
i = s.index("\ndef prepare(weights):")
j = s.index("\n    return state\n", i)
s = (s[:j] + "\n    try:\n        _diag_proj(state[\"dense\"])  # v26p: diagnostic only, untimed\n    except Exception as e:  # never fail the run\n"
     "        print(\"v26p diag failed:\", repr(e)[:500], flush=True)" + s[j:])
s = rep(s, '# okc: problem=muse_decode_layer\n"""v25:', '# okc: problem=muse_decode_layer\n"""v26p (DIAGNOSTIC: v25 + an untimed one-time projection sweep at the end of prepare(), prints only).\n\nv25:')
open(DST, "w").write(s)
print("wrote", DST)
