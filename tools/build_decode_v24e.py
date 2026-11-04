"""Build muse_decode_layer/runs/v24e_bench.py (DIAGNOSTIC): decode v24 (runs/v24_rows_occ4.py, kernel_fn unchanged)
plus a one-time untimed sweep at the end of prepare() of the CUDA experts' launch configs (SMALL_CFG: NB, and per GEMM
WARPS / split / prefetch) at 32 / 128 / 256 tokens with the judge's real weights, prints only. Routing: the layer's
own router + bias + SGLang's routing on random hidden rows (local experts 0..31). Cold L2 (256 MB read), median of
11; each config's output is compared with the current config's. Coordinate descent per token count: NB, then GEMM 1
(W, S, P), then GEMM 2. A time guard stops at 170 s. Run from the repo root.
"""
import sys

SRC = "muse_decode_layer/runs/v24_rows_occ4.py"
DST = "muse_decode_layer/runs/v24e_bench.py"
s = open(SRC).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


BENCH = r'''

# ---- v24e diagnostic (prepare() only; never on the timed path) ----
_DIAG = {}


def _diag_bench(weights, state):
    import time
    if _DIAG or not torch.cuda.is_available():
        return
    _DIAG["done"] = True
    t_start = time.time()
    dev = weights["qkv_gate"].device
    g = torch.Generator(device=dev).manual_seed(4321)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    st = state["moe_small"]
    H = weights["w2"].shape[1]
    I = weights["w13"].shape[1] // 2

    def timed(f, reps=11):
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

    def ok_cfg(nb, w, s_, gemm):
        k = (H if gemm == 1 else I) // s_
        return 8 * nb * (k + 8) * 2 <= 48 * 1024 and k % 64 == 0 and (I if gemm == 1 else H) % (16 * w) == 0

    print("v24e diag: MoE small-kernel sweep, H", H, "I", I, "fast", st["fast"], flush=True)
    for T in (256, 128, 32):
        x8k = torch.randn((T, HIDDEN), generator=g, device=dev, dtype=torch.float32)
        logits = F.linear(x8k, state["router"])
        tw, tid = state["routing"](None, logits, TOP_K, True)
        local = (tid.to(torch.int32) % N_EXPERTS)
        loc = local[local < LOCAL_EXPERTS]
        cnt = torch.bincount(loc.long(), minlength=LOCAL_EXPERTS)
        hs = torch.randn((T, H), generator=g, device=dev, dtype=torch.float32).to(torch.bfloat16)
        data = {"hidden_states": hs, "topk_ids": local, "topk_weights": tw,
                "config": {"d_expert": D_EXPERT, "top_k": TOP_K, "local_start": 0, "local_count": LOCAL_EXPERTS}}
        print(f"T {T}: local routes {int(loc.numel())} of {T * TOP_K}, active local experts {int((cnt > 0).sum())}, "
              f"max per expert {int(cnt.max())}", flush=True)
        base = _small_cfg(T)
        ref = _kernel_small(data, st, base).float()
        torch.cuda.synchronize()
        res = {}

        def run(cfg):
            if cfg in res:
                return res[cfg]
            if time.time() - t_start > 170:
                return float("inf")
            try:
                out = _kernel_small(data, st, cfg).float()
                torch.cuda.synchronize()
                err = ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()
                t = timed(lambda: _kernel_small(data, st, cfg))
            except Exception as e:
                print(f"  {cfg}: failed {repr(e)[:100]}", flush=True)
                res[cfg] = float("inf")
                return res[cfg]
            if not err < 1e-3:
                print(f"  {cfg}: WRONG err {err:.2e}", flush=True)
                t = float("inf")
            else:
                print(f"  {cfg}: {t:.1f} us  err {err:.1e}", flush=True)
            res[cfg] = t
            return t

        best = base
        run(base)
        for nb in (1, 2, 4):
            for g1 in ((8, 4, 2), (8, 8, 2), (4, 8, 2)):
                if ok_cfg(nb, g1[0], g1[1], 1) and ok_cfg(nb, g1[0], g1[1], 2):
                    c = (nb, g1, g1)
                    if run(c) < run(best):
                        best = c
        nb = best[0]
        for gemm in (1, 2):
            for w in (4, 8, 16):
                for s_ in (2, 4, 8):
                    for p in (2, 3):
                        if not ok_cfg(nb, w, s_, gemm):
                            continue
                        c = (nb, (w, s_, p), best[2]) if gemm == 1 else (nb, best[1], (w, s_, p))
                        if run(c) < run(best):
                            best = c
        print(f"T {T}: current {base} {res[base]:.1f} us -> best {best} {res[best]:.1f} us", flush=True)
    del flush
    torch.cuda.empty_cache()
    print(f"v24e diag: done in {time.time() - t_start:.0f} s", flush=True)

'''
s = rep(s, "\ndef prepare(weights):", BENCH + "\ndef prepare(weights):")
i = s.index("\ndef prepare(weights):")
j = s.index("\n    return state\n", i)
s = (s[:j] + "\n    try:\n        _diag_bench(weights, state)  # v24e: diagnostic only, untimed\n    except Exception as e:  # never fail the run\n"
     "        print(\"v24e diag failed:\", repr(e)[:500], flush=True)" + s[j:])
s = rep(s, '# okc: problem=muse_decode_layer\n"""v24:', '''# okc: problem=muse_decode_layer
"""v24e (DIAGNOSTIC: v24 + an untimed one-time CUDA-experts launch-config sweep at the end of prepare(), prints only;
tools/build_decode_v24e.py).

v24:''')
open(DST, "w").write(s)
print("wrote", DST)
