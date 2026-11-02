"""Build the attention v25 runs files from muse_attention_decode/runs/v24_rows_occ4.py (board 1.517x, submit
7e6f6a28c30b4688) + tools/attn_v25_fv.cu (v24's full-verify kernel source with the "memory" clobber on the V
transpose's inline ld/st.shared (review r17 m1) and ST overridable; default ST 3 = v24's cubin byte for byte).
Run from the repo root.

  python3 tools/build_attn_v25.py --diag [name]       -> runs/<name or v25d_bench>.py: v24's paths unchanged + an untimed
                                                         sweep at import (full verify b 16/32: FV ST x splits/SM; full
                                                         decode b 16/32: rows kernel BLOCK_N x warps x stages x splits/SM)
  python3 tools/build_attn_v25.py CFG.json name        -> runs/<name>.py with {"fv_st": n, "fv_occ": n,
                                                         "rows": [BLOCK_N, warps, stages, occ, hoist]}
"""
import json
import sys

SRC = "muse_attention_decode/runs/v24_rows_occ4.py"
s = open(SRC).read()
cu = open("tools/attn_v25_fv.cu").read()
assert '"""' not in cu and cu.count('"memory"') >= 16
ws = open("tools/attn_v25_ws.cu").read()  # warp-specialized variant (flag bit 6 of fv_vt1)
assert '"""' not in ws and ws.count('"memory"') >= 20
DIAG = sys.argv[1] == "--diag"
cfg = {"fv_st": 3, "fv_occ": 4, "fv_vt1": 0, "rows": [128, 4, 3, 4, True]}
if DIAG:
    name = sys.argv[2] if len(sys.argv) > 2 else "v25d_bench"
else:
    cfg.update(json.load(open(sys.argv[1])))
    name = sys.argv[2]


def rep(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


i = s.index('_FV_SRC = r"""') + len('_FV_SRC = r"""')
j = s.index('"""', i)
s = s[:i] + "\n" + cu + s[j:]
k = s.index("\n", i + 1 + len(cu)) + 1  # after the closing line of _FV_SRC
s = s[:k] + '_WS_SRC = r"""\n' + ws + '"""  # tools/attn_v25_ws.cu: attn_fvws (fv_vt1 & 64)\n' + s[k:]

s = rep(s, '''FV_BN, FV_ST = 64, 3  # must match BN / ST in tools/attn_v22_fv.cu
FV_SMEM = 2 * FV_ST * FV_BN * 128 + 2 * 2 * 9216 + 2 * 8192 + 1024  # K + V E4M3 stages, 2 V^T buffers (72 rows) and Q, alignment
FV_OCC = 4  # split-KV programs per SM for _nsplit (the kernel fits 2 per SM; the rest is load balance)''',
        f'''FV_BN = 64  # must match BN in tools/attn_v25_fv.cu
FV_ST = {cfg["fv_st"]}  # cp.async stages (-DST; 3 = v24)
FV_VT1 = {cfg["fv_vt1"]}  # flags: VT1 | FVSKIP << 1 (diag) | PF << 3 (3 bits) | 64 = warp-specialized attn_fvws; 0 = v24
FV_OCC = {cfg["fv_occ"]}  # split-KV programs per SM for _nsplit (the kernel fits 2 per SM at ST 3; the rest is load balance)
ROWS_FULL = ("rows", {cfg["rows"][0]}, {cfg["rows"][1]}, {cfg["rows"][2]}, {cfg["rows"][3]}, {cfg["rows"][4]})  # full decode, batch >= 2


def _fv_smem(st, vt1=None):
    vt1 = FV_VT1 if vt1 is None else vt1
    return 2 * st * FV_BN * 128 + (1 if vt1 & 1 else 2) * 2 * 9216 + 2 * 8192 + 1024 + (
        128 if vt1 & 64 else 0)  # K + V E4M3 stages, V^T buffers (72 rows), Q, alignment (+ WS mbarriers)


FV_SMEM = _fv_smem(FV_ST)''')

s = rep(s, '''def _fv_cubin():
    """NVRTC (cuda.bindings, in-process) -> sm_90a cubin bytes (no files). Host-only, no CUDA calls."""
    if "cubin" not in _FV:''', '''def _fv_cubin(st=None, vt1=None):
    """NVRTC (cuda.bindings, in-process) -> sm_90a cubin bytes (no files). Host-only, no CUDA calls."""
    st = FV_ST if st is None else st
    vt1 = FV_VT1 if vt1 is None else vt1
    if ("cubin", st, vt1) not in _FV:''')
s = rep(s, '''        opts = [b"-arch=sm_90a", b"-std=c++17"]''', '''        opts = [b"-arch=sm_90a", b"-std=c++17", f"-DST={st}".encode()] + ([b"-DVT1=1"] if vt1 & 1 else []) + (
            [f"-DFVSKIP={(vt1 >> 1) & 3}".encode()] if (vt1 >> 1) & 3 else []) + (
            [f"-DPF={(vt1 >> 3) & 7}".encode()] if (vt1 >> 3) & 7 else [])  # vt1 = VT1 | FVSKIP << 1 (diag) | PF << 3 | WS 64''')
s = rep(s, '''        _FV["cubin"] = cubin
    return _FV["cubin"]''', '''        _FV[("cubin", st, vt1)] = cubin
    return _FV[("cubin", st, vt1)]''')
s = rep(s, '''def _fv_fn(dev):''', '''def _fv_fn(dev, st=None, vt1=None):''')
s = rep(s, '''    if dev not in _FV:
        if torch.cuda.is_current_stream_capturing():''', '''    st = FV_ST if st is None else st
    vt1 = FV_VT1 if vt1 is None else vt1
    if (dev, st, vt1) not in _FV:
        if torch.cuda.is_current_stream_capturing():''')
s = rep(s, '''        cubin = _fv_cubin()''', '''        cubin = _fv_cubin(st, vt1)''')
s = rep(s, '''            drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8), FV_SMEM)
        _FV[dev] = (mod, f)
    return _FV[dev][1]''', '''            drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8), _fv_smem(st, vt1))
        _FV[(dev, st, vt1)] = (mod, f)
    return _FV[(dev, st, vt1)][1]''')
s = rep(s, '''def _fv(data, q, qkv, cfg):''', '''def _fv(data, q, qkv, cfg, st=None, occ=None, vt1=None):''')
s = rep(s, '''    nsplit = _nsplit(cfg, batch, 1, capacity, FV_OCC)''', '''    st = FV_ST if st is None else st
    vt1 = FV_VT1 if vt1 is None else vt1
    nsplit = _nsplit(cfg, batch, 1, capacity, FV_OCC if occ is None else occ)''')
s = rep(s, '''    f = _fv_fn(dev)
    with''', '''    f = _fv_fn(dev, st, vt1)
    with''')
s = rep(s, '''            np.int32(hq)), shared_mem=FV_SMEM)''', '''            np.int32(hq)), shared_mem=_fv_smem(st, vt1))''')
s = rep(s, '''    return ("rows", 128, 4, 3, 4, True)  # v24: ~4 programs/SM (was 16); 128-key tiles + PREF (= hoist), v20 / v17''',
        '''    return ROWS_FULL  # v24: ("rows", 128, 4, 3, 4, True): ~4 programs/SM, 128-key tiles + PREF (= hoist); v25 config''')
# kernel_fn: allow a path override (diagnostic sweeps call _kernel_path directly)
s = rep(s, '''def kernel_fn(data):
    cfg = data["config"]''', '''def kernel_fn(data):
    return _kernel_path(data)


def _kernel_path(data, path_override=None):
    cfg = data["config"]''')
s = rep(s, '''    path = _attn_path(kind, L, batch, hkv, d)
    if path is None and _fv_ok''', '''    path = _attn_path(kind, L, batch, hkv, d) if path_override is None else path_override
    if path is None and _fv_ok''')

s = rep(s, '''        err, prog = nvrtc.nvrtcCreateProgram(_FV_SRC.encode(), b"attn_fv.cu", 0, [], [])''',
        '''        err, prog = nvrtc.nvrtcCreateProgram((_WS_SRC if vt1 & 64 else _FV_SRC).encode(), b"attn_fv.cu", 0, [], [])''')
s = rep(s, '''            f = mod.get_function("attn_fv")''', '''            f = mod.get_function("attn_fvws" if vt1 & 64 else "attn_fv")''')
s = rep(s, '''        f((nsplit, batch), (256,), (''', '''        f((nsplit, batch), (288 if vt1 & 64 else 256,), (''')

head = ('''# okc: problem=muse_attention_decode
"""''' + name + ''': v24 + attention v25 source (tools/build_attn_v25.py): the full-verify kernel from tools/attn_v25_fv.cu
(v24's + the "memory" clobber on the V transpose's inline ld/st.shared, review r17 m1; ST at compile time), FV ST
''' + str(cfg["fv_st"]) + ''', VT1 ''' + str(cfg["fv_vt1"]) + ''', FV splits/SM ''' + str(cfg["fv_occ"]) + ''', full decode batch >= 2 rows config ''' + str(tuple(cfg["rows"])) + '''.
''' + ('''v25d (DIAGNOSTIC): kernel_fn is v24's; at import (GPU present) an untimed sweep of full verify (b 16/32) and full
decode (b 16/32) variants on synthetic agentic inputs (ctx 80000, page 1, scattered pages), CUDA graphs, cold L2.
''' if DIAG else "") + '''
v24:''')
s = rep(s, '''# okc: problem=muse_attention_decode
"""v24:''', head)

if DIAG:
    s += r'''


_DIAG25 = {}


def _v25_inputs(batch, L, seed, dev, ctx=80000):
    """generate_input-like synthetic inputs: agentic lengths (ctx +/- 7.5%), page 1, scattered pages, N(0,1) FP8 K/V."""
    g = torch.Generator(device=dev).manual_seed(seed)
    hq, hkv, d = 16, 2, 64
    u = torch.rand((batch,), generator=g, device=dev)
    seq = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, int(ctx * 1.075) + 1)
    longest = int(ctx * 1.075) + 1
    maxp = longest + L
    total = batch * maxp
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    pt = perm.view(batch, maxp).contiguous()
    kc = torch.randn((total, 1, hkv, d), generator=g, device=dev).to(FP8)
    vc = torch.randn((total, 1, hkv, d), generator=g, device=dev).to(FP8)
    qkv = torch.randn((batch * L, (hq + 2 * hkv) * d + hq * d), generator=g, device=dev).to(torch.bfloat16)
    positions = (seq[:, None] + torch.arange(L, device=dev)).reshape(-1)
    pos = positions.view(batch, L)
    loc = torch.gather(pt.long(), 1, pos).reshape(-1)
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    return {"qkv": qkv, "positions": positions, "cache_loc": loc, "cache_seqlens": (seq + L).to(torch.int32),
            "cu_seqlens_q": torch.arange(0, batch * L + 1, L, dtype=torch.int32, device=dev),
            "scheduler_metadata": None, "k_cache": kc, "v_cache": vc, "k_scale": one, "v_scale": one,
            "page_table": pt, "seq_lens": seq.to(torch.int32),
            "config": {"batch": batch, "q_len": L, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d,
                       "page_size": 1, "window": 0, "rope_theta": 0.0, "qk_scale_factor": 5.4730064863875,
                       "sm_scale": d ** -0.5, "kv_scale": 1.0}}


def _v25_case(batch, ctx, page, spread, seed, dev, kv_scale=1.0, L=8):
    """Validation inputs (no reference import: the server rejects it): page size `page`, scattered pages,
    lengths fixed / agentic (+/- 7.5%) / mixed (uniform 1 .. ctx), FP8 K/V with scale kv_scale, -1 past the end."""
    g = torch.Generator(device=dev).manual_seed(seed)
    hq, hkv, d = 16, 2, 64
    u = torch.rand((batch,), generator=g, device=dev)
    if spread == "agentic":
        seq = (ctx * (1 + 0.075 * (2 * u - 1))).long()
    elif spread == "mixed":
        seq = (1 + u * (ctx - 1)).long()
    else:
        seq = torch.full((batch,), ctx, device=dev, dtype=torch.long)
    seq = seq.clamp(1, int(ctx * 1.075) + 1)
    longest = int(ctx * 1.075) + 1
    maxp = (longest + L + page - 1) // page
    total = batch * maxp
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    pt = torch.full((batch, maxp), -1, dtype=torch.int32, device=dev)
    need = (seq + L + page - 1) // page
    for bb in range(batch):
        pt[bb, :int(need[bb])] = perm[bb * maxp: bb * maxp + int(need[bb])]
    kc = (torch.randn((total, page, hkv, d), generator=g, device=dev) / kv_scale).to(FP8)
    vc = (torch.randn((total, page, hkv, d), generator=g, device=dev) / kv_scale).to(FP8)
    qkv = torch.randn((batch * L, (hq + 2 * hkv) * d + hq * d), generator=g, device=dev).to(torch.bfloat16)
    positions = (seq[:, None] + torch.arange(L, device=dev)).reshape(-1)
    pos = positions.view(batch, L)
    loc = (torch.gather(pt.long(), 1, pos // page) * page + pos % page).reshape(-1)
    sc = torch.tensor(kv_scale, dtype=torch.float32, device=dev)
    return {"qkv": qkv, "positions": positions, "cache_loc": loc, "cache_seqlens": (seq + L).to(torch.int32),
            "cu_seqlens_q": torch.arange(0, batch * L + 1, L, dtype=torch.int32, device=dev),
            "scheduler_metadata": None, "k_cache": kc, "v_cache": vc, "k_scale": sc, "v_scale": sc.clone(),
            "page_table": pt, "seq_lens": seq.to(torch.int32),
            "config": {"batch": batch, "q_len": L, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d,
                       "page_size": page, "window": 0, "rope_theta": 0.0, "qk_scale_factor": 5.4730064863875,
                       "sm_scale": d ** -0.5, "kv_scale": kv_scale}}


WS_EXACT_TOL = 0.0   # max |ws - v24| at the same split count / ST (same smem bytes, same MMA and softmax order)
WS_SPLIT_TOL = 2e-2  # rel-L2 of any (ST, splits) output vs v24's default config (split order changes bf16 rounding)


def _v25_validate():
    """WS kernel (flags 64) vs v24's FV kernel (judge-validated) at the same ST and split count: bit-exact expected
    (assert max |diff| <= WS_EXACT_TOL); every output finite and within WS_SPLIT_TOL rel-L2 of v24 at its default
    config. Cases: local-test-like full q_len 8 shapes (pages 1 / 16 / 64, fixed / agentic / mixed lengths, tiny
    contexts with empty splits, kv_scale 1/64) x forced splits/SM 1 .. 33 x ST 2 / 3."""
    import time
    t0 = time.time()
    dev = torch.cuda.current_device()
    cases = [(2, 900, 1, "fixed", 0.015625), (3, 3000, 64, "fixed", 1.0), (2, 3000, 64, "fixed", 1.0),
             (17, 1800, 64, "agentic", 1.0), (2, 700, 1, "mixed", 1.0), (2, 10, 64, "fixed", 1.0),
             (2, 10, 1, "fixed", 1.0), (4, 2500, 1, "mixed", 1.0), (3, 2000, 16, "agentic", 1.0),
             (5, 4100, 1, "agentic", 1.0), (7, 130, 2, "mixed", 1.0), (16, 80000, 1, "agentic", 1.0)]
    n = bad = 0
    for ci, (batch, ctx, page, spread, kvs) in enumerate(cases):
        if time.time() - t0 > 120:
            print("v25 WS validate: (time guard)", flush=True)
            break
        data = _v25_case(batch, ctx, page, spread, 500 + ci, dev, kvs)
        cfg = data["config"]
        T = batch * 8

        def run(st, occ, flags):
            qkv = data["qkv"].view(T, -1)
            q = _norm_rope_store_fused(data, qkv, cfg, FP8)
            return _fv(data, q, qkv, cfg, st, occ, flags).float()
        base = run(FV_ST, FV_OCC, 0)
        for occ in ((1, 2, 4, 8, 33) if batch < 16 else (2, 4)):
            for st in (3, 2):
                v24o, wso = run(st, occ, 0), run(st, occ, 64)
                torch.cuda.synchronize()
                dmax = float((wso - v24o).abs().max())
                rel = float((wso - base).norm() / base.norm().clamp_min(1e-30))
                fin = bool(torch.isfinite(wso).all())
                n += 1
                try:
                    assert fin, "non-finite"
                    assert dmax <= WS_EXACT_TOL, f"max|ws-v24| {dmax:.3e} > {WS_EXACT_TOL}"
                    assert rel <= WS_SPLIT_TOL, f"rel-L2 vs v24 default {rel:.2e} > {WS_SPLIT_TOL}"
                except AssertionError as e:
                    bad += 1
                    print(f"  WS FAIL b {batch} ctx {ctx} page {page} {spread} kvs {kvs} occ {occ} st {st}: {e}", flush=True)
        del data
    print(f"v25 WS validate: {'PASS' if bad == 0 and n else 'FAIL'} {n - bad}/{n} (bit-exact vs v24 FV at equal splits,"
          f" rel-L2 <= {WS_SPLIT_TOL} vs v24 default) in {time.time() - t0:.0f} s", flush=True)
    return bad == 0 and n > 0


def _v25_diag():
    import time
    if _DIAG25 or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return
    _DIAG25["done"] = True
    t0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device())
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    tiny = torch.empty(16, device=dev)

    def timed(fn, reps=11):
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        ts = []
        for _ in range(reps):
            flush.view(torch.int32).sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    floor = timed(lambda: tiny.zero_())
    print(f"v25d diag: {torch.cuda.get_device_name()} floor {floor:.2f} us", flush=True)
    try:
        _v25_validate()
    except Exception as e:
        print("v25 WS validate: ERROR", repr(e)[:400], flush=True)
    summ = []
    for batch, L in ((16, 8), (32, 8)):
        if time.time() - t0 > 240:
            print("(time guard)", flush=True)
            break
        data = _v25_inputs(batch, L, 100 + batch + L, dev)
        nbytes = int(data["cache_seqlens"].long().sum()) * 256
        ref = kernel_fn(data).float()
        tb = timed(lambda: kernel_fn(data))
        res = []
        if L == 8:
            cands = [(3, 4, 0), (3, 4, 64), (3, 2, 64), (3, 3, 64), (3, 6, 64), (3, 8, 64), (2, 4, 64), (2, 2, 64)]  # (ST, splits/SM, flags; 64 = WS)
            for st, occ, vt1 in cands:
                def fn(st=st, occ=occ, vt1=vt1):
                    cfg = data["config"]
                    T = batch * L
                    qkv = data["qkv"].view(T, -1)
                    q = _norm_rope_store_fused(data, qkv, cfg, FP8)
                    return _fv(data, q, qkv, cfg, st, occ, vt1)
                try:
                    o = fn().float()
                    err = float((o.view(-1) - ref.view(-1)).norm() / ref.norm())
                    res.append((timed(fn), err, ("fv", st, occ, vt1)))
                except Exception as e:
                    print(f"  b {batch} L {L} fv {st} {occ} {vt1} FAILED {repr(e)[:200]}", flush=True)
        else:
            cands = [(128, 4, 3, 2)]
            for bn, w, stg, occ in cands:
                if True:
                    if True:
                        if True:
                            p = ("rows", bn, w, stg, occ, True)
                            try:
                                o = _kernel_path(data, p).float()
                                err = float((o.view(-1) - ref.view(-1)).norm() / ref.norm())
                                res.append((timed(lambda p=p: _kernel_path(data, p)), err, p[1:]))
                            except Exception as e:
                                print(f"  b {batch} L {L} {p} FAILED {repr(e)[:160]}", flush=True)
        res.sort()
        print(f"b {batch} L {L}: {nbytes / 1e6:.0f} MB K+V, v24 {tb:.1f} us ({nbytes / max(tb - floor, 1e-3) / 1e6:.2f} TB/s)"
              f" | best {res[0][0]:.1f} us {res[0][2]} e{res[0][1]:.1e}", flush=True)
        print("  " + "; ".join(f"{c} {t:.1f} e{e:.0e}" for t, e, c in res[:12]), flush=True)
        summ.append((batch, L, tb, res[0][0], res[0][2]))
        del data
    print("v25d summary (b L: v24 / best us, cfg): " + "; ".join(f"{b} {l}: {a:.1f} / {c:.1f} {p}" for b, l, a, c, p in summ),
          flush=True)
    print(f"v25d diag: done in {time.time() - t0:.0f} s", flush=True)


try:
    _v25_diag()  # v25d: diagnostic only, untimed (at import, before the judge's timing)
except Exception as _e:  # never fail the run
    print("v25d diag failed:", repr(_e)[:600], flush=True)
'''

dst = f"muse_attention_decode/runs/{name}.py"
open(dst, "w").write(s)
print("wrote", dst)
