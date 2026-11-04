"""Build decode v24 from runs/v23_fv_attn.py (check 7f0858a69e0b4492, 1.376x). Run from the repo root.

  python3 tools/build_decode_v24.py          -> muse_decode_layer/runs/v24_rows_occ4.py   (candidate)
  python3 tools/build_decode_v24.py --diag   -> muse_decode_layer/runs/v24d_bench.py      (same kernel_fn + an
                                                untimed one-time sweep at the end of prepare(), prints only)

v24 = v23 with full-attention decode (q_len 1) at batch >= 2 on the split-KV "rows" kernel with 128-key tiles,
~4 programs per SM (was 64-key tiles, ~16 per SM) and the next tile's page indices loaded one iteration ahead
(PREF; _attn_split_rows replaced by attention v23's, identical apart from PREF). Evidence: diagnostic check
6567d26179874f80's in-prepare sweep on the judge's H200 (cold L2): b16 148.8 -> 127.6 us, b32 263.0 -> 234.7 us.
"""
import sys

DIAG = "--diag" in sys.argv
SRC = "muse_decode_layer/runs/v23_fv_attn.py"
ATT = "muse_attention_decode/runs/v23_fv_addr.py"
DST = "muse_decode_layer/runs/v24d_bench.py" if DIAG else "muse_decode_layer/runs/v24_rows_occ4.py"
s = open(SRC).read()
a = open(ATT).read()


def rep(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


def fn(text, name):
    i = text.index("def %s(" % name)
    j = text.index("\n@triton.jit", i)
    return text[i:j]


s = rep(s, fn(s, "_attn_split_rows"), fn(a, "_attn_split_rows"))
s = rep(s, '''    if batch == 1:
        return ("rows", 64, 4, 3, 4, False)
    return ("rows", 64, 4, 3, 16, True)''', '''    if batch == 1:
        return ("rows", 64, 4, 3, 4, False)
    return ("rows", 128, 4, 3, 4, True)  # v24: 128-key tiles, ~4 programs/SM, PREF (check 6567d26179874f80)''')
s = rep(s, "#   full decode, batch >= 2                  \"rows\", ~16 programs/SM, V load hoisted: 1.17-1.26x\n"
           "#                                            at b 16/32 (the 4/SM \"head\" config was 0.92x there)",
        "#   full decode, batch >= 2                  \"rows\", 128-key tiles, ~4 programs/SM, V load hoisted, next\n"
        "#                                            tile's page indices prefetched (v24; v19's 64-key ~16/SM config\n"
        "#                                            measured 148.8 / 263.0 us vs 127.6 / 234.7 at b16 / b32, check 6567d261)")
s = rep(s, '''            HW=True, HOIST=hoist, PV16=True, num_warps=warps, num_stages=stages)
    else:''', '''            HW=True, HOIST=hoist, PV16=True, PREF=hoist, num_warps=warps, num_stages=stages)
    else:''')
hdr = '''# okc: problem=muse_decode_layer
"""v24: v23 with full-attention decode (q_len 1) at batch >= 2 on the split-KV "rows" kernel with 128-key tiles at
~4 programs per SM (v19-v23: 64-key tiles, ~16 per SM) and the next tile's page indices loaded one iteration ahead
(PREF: attention v23's _attn_split_rows). Diagnostic check 6567d26179874f80's in-prepare sweep on the judge's H200
(cold L2, ctx 80000, page 1, agentic): rows + combine 148.8 -> 127.6 us at batch 16 and 263.0 -> 234.7 us at 32.
tools/build_decode_v24.py.

v23:'''
if DIAG:
    hdr = hdr.replace('"""v24:', '"""v24d (DIAGNOSTIC: v24 + an untimed one-time sweep at the end of prepare(), prints only).\n\nv24:')
s = rep(s, '# okc: problem=muse_decode_layer\n"""v23:', hdr)

if DIAG:
    BENCH = r'''

# ---- v24d diagnostic (prepare() only; never on the timed path) ----
_DIAG = {}


def _diag_bench():
    """One-time prepare() sweep on synthetic inputs (ctx 80000 +/- 7.5%, page 1, scattered pages, cold L2 by reading a
    256 MB buffer, median of 15): (1) full decode rows kernel configs at batch 16 / 32, (2) the full-verify CUDA kernel
    (attn_fv) at batch 16 / 32 over programs per SM (FV_OCC), (3) sliding decode / verify head kernel at batch 32."""
    import time
    if _DIAG or not torch.cuda.is_available():
        return
    _DIAG["done"] = True
    import cupy as cp
    t_start = time.time()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1234)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    hq, hkv, d, W = 16, 2, 64, 128

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

    print("v24d diag:", torch.cuda.get_device_name(), "triton", triton.__version__, flush=True)
    for batch in (16, 32):
        ctx = 80000
        u = torch.rand((batch,), generator=g, device=dev)
        lens = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, math.ceil(ctx * 1.075))
        cap = math.ceil(ctx * 1.075) + 8
        total = batch * cap
        pt = torch.randperm(total, generator=g, device=dev).to(torch.int32).view(batch, cap).contiguous()
        kc = (torch.randint(0, 256, (total, W), generator=g, device=dev, dtype=torch.int32) & 0x77).to(torch.uint8)
        vc = (torch.randint(0, 256, (total, W), generator=g, device=dev, dtype=torch.int32) & 0x77).to(torch.uint8)
        one = torch.ones((), dtype=torch.float32, device=dev)
        for L in (1, 8):
            T = batch * L
            sl = (lens + L).to(torch.int32)
            nbytes = 2 * W * int(sl.sum())
            q = (2 * torch.randn((T, hq * d), generator=g, device=dev)).to(FP8)
            gate = torch.randn((T, hq * d), generator=g, device=dev).to(torch.bfloat16)
            cfg = {"window": 0, "q_len": L, "batch": batch, "page_size": 1, "num_q_heads": hq, "num_kv_heads": hkv,
                   "head_dim": d, "sm_scale": 0.125, "qk_scale_factor": 1.0}
            print(f"batch {batch} q_len {L} full: K+V {nbytes / 1e6:.1f} MB (4.8 TB/s floor {nbytes / 4.8e6:.1f} us)",
                  flush=True)
            if L == 1:
                confs = [(128, 4, 3, 4, True)]
                for occ in (2, 3, 4, 6):
                    for bn, wp in ((64, 2), (128, 2), (128, 4), (256, 4), (64, 1), (128, 1)):
                        for st in (2, 3, 4):
                            confs.append((bn, wp, st, occ, True))
                confs += [(128, 4, 3, 4, False), (128, 2, 3, 4, False), (64, 2, 3, 16, True), (64, 2, 2, 16, True)]
                ref = None
                for bn, wp, st, occ, hoist in confs:
                    if time.time() - t_start > 200:
                        print("  (time guard)", flush=True)
                        break
                    nsp = _nsplit(cfg, batch, 1, cap, occ)
                    chunk = triton.cdiv(cap, nsp)
                    o_part = torch.empty((T * hq * nsp, d), device=dev, dtype=torch.float32)
                    lse = torch.empty((T * hq * nsp,), device=dev, dtype=torch.float32)
                    out = torch.empty((T, hq * d), device=dev, dtype=torch.bfloat16)

                    def f():
                        _attn_split_rows[(nsp, batch)](
                            q, q.stride(0), kc, vc, pt, pt.stride(0), sl, one, one, o_part, lse, 0.125, chunk, 1,
                            L=1, G=8, HKV=2, NSPLIT=nsp, WINDOW=0, BLOCK_M=16, BLOCK_N=bn, D=d, HW=True, HOIST=hoist,
                            PV16=True, PREF=hoist, num_warps=wp, num_stages=st)
                        _combine_norm_gate[(T, hq)](o_part, lse, gate, gate.stride(0), out, RMS_EPS, HQ=hq,
                                                    NSPLIT=nsp, D=d, num_warps=4 if nsp >= 64 else 1)
                    try:
                        f()
                        torch.cuda.synchronize()
                    except Exception as e:
                        print(f"  rows BN {bn} w {wp} st {st} occ {occ}: failed {repr(e)[:120]}", flush=True)
                        continue
                    if ref is None:
                        ref = out.float().clone()
                    err = ((out.float() - ref).norm() / ref.norm()).item()
                    t = timed(f)
                    print(f"  rows BN {bn} w {wp} st {st} occ {occ} (nsplit {nsp}) hoist/pref {int(hoist)}: {t:.1f} us "
                          f"{nbytes / t / 1e6:.2f} TB/s err {err:.1e}", flush=True)
            else:
                fvf = _fv_fn(torch.cuda.current_device())
                ref = None
                for occ in (1, 2, 3, 4, 6, 8, 12):
                    if time.time() - t_start > 230:
                        print("  (time guard)", flush=True)
                        break
                    nsp = _nsplit(cfg, batch, 1, cap, occ)
                    chunk = triton.cdiv(cap, nsp)
                    o_part = torch.empty((T * hq * nsp, d), device=dev, dtype=torch.float32)
                    lse = torch.empty((T * hq * nsp,), device=dev, dtype=torch.float32)
                    out = torch.empty((T, hq * d), device=dev, dtype=torch.bfloat16)

                    def f():
                        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                            fvf((nsp, batch), (256,), (
                                np.uint64(q.data_ptr()), np.int32(q.stride(0)), np.uint64(kc.data_ptr()),
                                np.uint64(vc.data_ptr()), np.uint64(pt.data_ptr()), np.int32(pt.stride(0)),
                                np.uint64(sl.data_ptr()), np.uint64(one.data_ptr()), np.uint64(one.data_ptr()),
                                np.uint64(o_part.data_ptr()), np.uint64(lse.data_ptr()), np.float32(0.125),
                                np.int32(chunk), np.int32(0), np.int32(nsp), np.int32(hq)), shared_mem=FV_SMEM)
                        _combine_norm_gate[(T, hq)](o_part, lse, gate, gate.stride(0), out, RMS_EPS, HQ=hq,
                                                    NSPLIT=nsp, D=d, num_warps=4 if nsp >= 64 else 1)
                    f()
                    torch.cuda.synchronize()
                    if ref is None:
                        ref = out.float().clone()
                    err = ((out.float() - ref).norm() / ref.norm()).item()
                    t = timed(f)
                    print(f"  attn_fv occ {occ} (nsplit {nsp}): {t:.1f} us {nbytes / t / 1e6:.2f} TB/s err {err:.1e}",
                          flush=True)
        del kc, vc, pt
        torch.cuda.empty_cache()
    del flush
    torch.cuda.empty_cache()
    print(f"v24d diag: done in {time.time() - t_start:.0f} s", flush=True)

'''
    s = rep(s, "\ndef prepare(weights):", BENCH + "\ndef prepare(weights):")
    i = s.index("\ndef prepare(weights):")
    j = s.index("\n    return state\n", i)
    s = (s[:j] + "\n    try:\n        _diag_bench()  # v24d: diagnostic only, untimed\n    except Exception as e:  # never fail the run\n"
         "        print(\"v24d diag failed:\", repr(e)[:500], flush=True)" + s[j:])
    s = rep(s, "\nimport types\n", "\nimport math\nimport types\n")
open(DST, "w").write(s)
print("wrote", DST)
