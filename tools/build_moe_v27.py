"""Builds the MoE v27 runs files from muse_moe_w4a16/runs/v26e_nb2_evict.py (exact replacements, each asserted once).
  python3 tools/build_moe_v27.py diag                 -> runs/v27d_bench.py  (kernel_fn = v26e; untimed v27 sweep in prepare)
  python3 tools/build_moe_v27.py route ROUTE.json NAME -> runs/NAME.py       (moe_stream for the token counts in ROUTE.json)
ROUTE.json: {"1": [[NT, NW, ST, TPS], [NT, NW, ST, TPS]], ...} = GEMM-1 and GEMM-2 tunes per scored token count (<= 32).
Run at HEAD; the committed files must be byte-identical to its output."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_moe_w4a16/runs/v26e_nb2_evict.py").read_text()
cu = (ROOT / "tools/moe_v27_stream.cu").read_text()
cfgsrc = (ROOT / "tools/moe_v27_cfg.py").read_text()
mode = sys.argv[1]
assert mode in ("diag", "route")
DIAG = mode == "diag"
if DIAG:
    name, route = "v27d_bench", {}
else:
    route = {int(k): (tuple(v[0]), tuple(v[1])) for k, v in json.loads(Path(sys.argv[2]).read_text()).items()}
    name = sys.argv[3]
    assert route and all(b <= 32 for b in route)
assert "'''" not in cu and '"""' not in cu


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


head = (f'''"""{name} = v26e + MoE v27 (tools/moe_v27_stream.cu): a weight-streaming grouped expert GEMM for <= 32 tokens.
v26e streams the touched experts at ~2.2-2.9 TB/s (moe_mma: per-warp register prefetch, split-K FP32 atomics).
v27: persistent grid (one CTA per SM), equal contiguous shares of the (segment x rowtile x k-chunk) stage sequence,
a producer warp streaming the A-fragment tiles with TMA bulk copies (L2 evict-first) into an mbarrier ring, the
segment's activations staged once in shared memory, no split-K atomics (the <= 2 units a CTA shares with its
neighbour are combined through a workspace + per-CTA launch counters; fixed order). GEMM 1 writes the BF16
SiLU product, GEMM 2 FP32 rows; route_prep (lists only, one block) and v26e's post-norm Triton kernel unchanged.
''' + ('''v27d: kernel_fn is v26e unchanged; prepare() runs a one-time untimed sweep of v27 tunes vs v26e (diagnostic).
''' if DIAG else f'''Routed token counts (GEMM-1 tune, GEMM-2 tune as (NT, NW, ST, TPS)): {sorted(route.items())}; other counts: v26e.
''') + '''
v26e = v26c with''')
rep('"""v26e = v26c with', head)

HOST = '''
# ---------------- MoE v27 (tools/moe_v27_stream.cu, configs tools/moe_v27_cfg.py) ----------------
_CU27 = r"""''' + cu + '''"""

''' + cfgsrc.replace('"""MoE v27 launch configs', '# """MoE v27 launch configs', 1).replace(
    "SMEM_LIMIT =", "V27_SMEM_LIMIT =").replace("SMEM_MIN =", "V27_SMEM_MIN =").replace(
    "smem(cfg, K) <= SMEM_LIMIT", "v27_smem(cfg, K) <= V27_SMEM_LIMIT").replace("def smem(", "def v27_smem(").replace(
    "def valid(", "def v27_valid(").replace("TUNES =", "V27_TUNES =").replace("in TUNES", "in V27_TUNES").replace(
    "if valid(", "if v27_valid(").replace("ALL_CONFIGS =", "V27_ALL =") + '''
V27_ROUTE = ''' + repr(route) + '''
_MODS27 = {}


def _mod27(cfg, fast, K):
    """cfg = (GATED, NT, NW, ST, TPS) -> (cupy function, dynamic smem bytes); compiled + attribute set once."""
    import cupy as cp
    key = (cfg, fast, K)
    if key not in _MODS27:
        g, nt, nw, st, tps = cfg
        assert v27_valid(cfg, K), cfg
        m = cp.RawModule(code=_CU27, options=("-std=c++17", f"-DGATED={g}", f"-DNT={nt}", f"-DNW={nw}", f"-DST={st}",
                                              f"-DTPS={tps}", f"-DFASTSCALE={fast}"))
        f = m.get_function("moe_stream")
        shm = max(v27_smem(cfg, K), V27_SMEM_MIN)  # >= 116 KB: one CTA per SM
        f.max_dynamic_shared_size_bytes = shm
        _MODS27[key] = (f, shm, nw)
    return _MODS27[key]


def _kernel_stream(data, state, t1, t2, stages=3):
    """t1 / t2 = GEMM-1 / GEMM-2 tunes (NT, NW, ST, TPS). stages < 3: diagnostic partial paths (route_prep + GEMM 1 ...)."""
    import cupy as cp
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    bs, H = x.shape
    I = cfg["d_expert"]
    top_k = cfg["top_k"]
    start, count = cfg["local_start"], cfg["local_count"]
    assert count == 32 and H % 1024 == 0 and I % 1024 == 0
    NR = bs * top_k
    MAXM = NR
    dev = x.device
    ib = torch.empty((NR + count + count * MAXM + NR + 1,), device=dev, dtype=torch.int32)
    slot = ib[:NR]
    cnt = ib[NR:NR + count]
    lst = ib[NR + count:NR + count + count * MAXM]
    work = ib[NR + count + count * MAXM:NR + count + count * MAXM + NR]
    nw = ib[NR + count + count * MAXM + NR:]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    inter = torch.empty((NR, I), device=dev, dtype=torch.bfloat16)
    y32 = torch.empty((NR, H), device=dev, dtype=torch.float32)  # non-local rows never read (slot < 0)
    f1, s1, w1 = _mod27((1,) + tuple(t1), state["fast"], H)
    f2, s2, w2 = _mod27((0,) + tuple(t2), state["fast"], I)
    P = state["v27_P"]
    ws, ctr = state["v27_ws"], state["v27_ctr"]
    m1 = _mod(4, H // 8, 1, 2, 1, state["fast"])  # route_prep only (lists, slot ids; RW1 = RW2 = 0: no zeroing)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((1,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                     np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(8),
                                                     np.uint64(0), np.int32(0), np.uint64(0), np.int32(0)))
        if stages >= 2:
            f1((P,), (32 * (w1 + 1),), (_p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(cnt),
                                       _p(lst), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H),
                                       np.int32(top_k), _p(inter), np.uint64(0), _p(ws), _p(ctr)), shared_mem=s1)
        if stages >= 3:
            f2((P,), (32 * (w2 + 1),), (_p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(cnt),
                                       _p(lst), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I),
                                       np.int32(1), np.uint64(0), _p(y32), _p(ws), _p(ctr)), shared_mem=s2)
    out = torch.empty((bs, H), device=dev, dtype=x.dtype)
    if stages >= 3:
        _post_norm_sum_kernel[(bs,)](y32, state["norm_w"], data["topk_weights"].float().contiguous(), slot, out,
                                     POST_NORM_EPS, TOPK=top_k, H=H, num_warps=8)
    return out


def _v27_prepare(st, H, I):
    dev = st["w13"].device
    P = torch.cuda.get_device_properties(dev).multi_processor_count
    st["v27_P"] = P
    st["v27_ws"] = torch.empty((P * 2 * 256,), device=dev, dtype=torch.float32)  # shared-unit partials
    st["v27_ctr"] = torch.zeros((P,), device=dev, dtype=torch.int32)  # per-CTA launch counters (lockstep)
    for t1, t2 in V27_ROUTE.values():  # NVRTC-compile every routed config now
        _mod27((1,) + tuple(t1), st["fast"], H)
        _mod27((0,) + tuple(t2), st["fast"], I)
'''

DIAGSRC = r'''

_DIAG27 = {}


def _v27_gen(bs, seed, dev):
    """reference.generate_input(bs, "ep8", seed, dist="normal", rank="busiest") routing, inlined (diagnostic only)."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def _v27_diag(state):
    """One-time untimed sweep per scored token count <= 32: v26e (as checked) vs v27 with every valid GEMM-1 tune
    (GEMM 2 default) then every GEMM-2 tune (best GEMM 1); CUDA graph replay, cold L2 (256 MB read first), median
    of 15; also route_prep + GEMM 1 alone and an empty-ish floor; rel L2 error of each v27 output vs v26e."""
    import time
    if _DIAG27 or not torch.cuda.is_available():
        return
    _DIAG27["done"] = True
    t0 = time.time()
    dev = state["w13"].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    tiny = torch.empty(16, device=dev)

    def timed(fn, reps=15):
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
    print(f"v27d diag: {torch.cuda.get_device_name()} SMs {state['v27_P']} fast {state['fast']} floor {floor:.2f} us",
          flush=True)
    EXP = 4096 * 4096 * 3 * 9 // 16  # bytes per expert (codes + scales), GEMM 1 is 2/3
    summ = []
    for bs in (1, 8, 16, 32):
        if time.time() - t0 > 420:
            print("(time guard)", flush=True)
            break
        data = _v27_gen(bs, 1000 + bs, dev)
        touched = int(torch.unique(data["topk_ids"][data["topk_ids"] < 32]).numel())
        ref = _kernel_small(data, state).float()
        t26 = timed(lambda: _kernel_small(data, state))
        nts = (1,) if bs <= 8 else (1, 2)
        res = []

        def run(t1, t2, stages=3):
            try:
                o = _kernel_stream(data, state, t1, t2, stages)
                if stages == 3:
                    torch.cuda.synchronize()
                    err = float((o.float() - ref).norm() / ref.norm().clamp(min=1e-30))
                else:
                    err = 0.0
                t = timed(lambda: _kernel_stream(data, state, t1, t2, stages))
                return t, err
            except Exception as e:
                print(f"  bs {bs} {t1} {t2} FAILED {repr(e)[:160]}", flush=True)
                return 1e9, 1.0

        d2 = (1, 8, 4, 16)
        g1 = []
        for nt in nts:
            for (nw, st, tps) in V27_TUNES:
                t1 = (nt, nw, st, tps)
                if v27_valid((1,) + t1, 4096) and v27_valid((0, nt) + d2[1:], 4096):
                    t, e = run(t1, (nt,) + d2[1:])
                    g1.append((t, e, t1))
        g1.sort()
        best1 = g1[0][2]
        g2 = []
        for (nw, st, tps) in V27_TUNES:
            t2 = (best1[0], nw, st, tps)
            if v27_valid((0,) + t2, 4096):
                t, e = run(best1, t2)
                g2.append((t, e, t2))
        g2.sort()
        best2 = g2[0][2]
        tg1, _ = run(best1, best2, 2)
        trp, _ = run(best1, best2, 1)
        tb = g2[0][0]
        bw1 = touched * EXP * 2 / 3 / max(tg1 - trp, 1e-3) / 1e6
        bw2 = touched * EXP / 3 / max(tb - tg1 - 3.0, 1e-3) / 1e6
        print(f"bs {bs}: experts {touched}, v26e {t26:.1f} us | v27 best {tb:.1f} us {best1} {best2} e{g2[0][1]:.1e}"
              f" | route_prep {trp:.1f}, +G1 {tg1:.1f} (~{bw1:.2f} TB/s G1, ~{bw2:.2f} TB/s G2 est)", flush=True)
        print("  G1 tunes: " + "; ".join(f"{c} {t:.1f} e{e:.0e}" for t, e, c in g1), flush=True)
        print("  G2 tunes: " + "; ".join(f"{c} {t:.1f} e{e:.0e}" for t, e, c in g2), flush=True)
        summ.append((bs, t26, tb, best1, best2))
    print("v27d summary (bs: v26e / v27 us, tunes): " + "; ".join(
        f"{b}: {a:.1f} / {c:.1f} {list(t1)} {list(t2)}" for b, a, c, t1, t2 in summ), flush=True)
    print(f"v27d diag: done in {time.time() - t0:.0f} s", flush=True)
'''

rep("\ndef prepare(weights):", HOST + (DIAGSRC if DIAG else "") + "\n\ndef prepare(weights):")
rep('''        _mod(W2, I // S2, 0, P2, NB, fast).compile()
    torch.cuda.synchronize()
    return st''', '''        _mod(W2, I // S2, 0, P2, NB, fast).compile()
    _mod(4, H // 8, 1, 2, 1, fast).compile()  # v27's route_prep module
    _v27_prepare(st, H, I)
    torch.cuda.synchronize()
''' + ('''    try:
        _v27_diag(st)  # v27d: diagnostic only, untimed
    except Exception as e:  # never fail the run
        print("v27d diag failed:", repr(e)[:600], flush=True)
''' if DIAG else "") + '''    return st''')
if not DIAG:
    rep('''    if data["hidden_states"].shape[0] <= SMALL_MAX:
        return _kernel_small(data, state)''', '''    bs = data["hidden_states"].shape[0]
    if bs in V27_ROUTE:
        return _kernel_stream(data, state, *V27_ROUTE[bs])
    if bs <= SMALL_MAX:
        return _kernel_small(data, state)''')

dst = ROOT / f"muse_moe_w4a16/runs/{name}.py"
dst.write_text(s)
print("wrote", dst)
