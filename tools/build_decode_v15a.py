"""Builds muse_decode_layer/runs/v15a_moe_bench.py (CHECK-ONLY exploration, never a submit) from v14_verify_split.py:
kernel_fn unchanged; prepare() additionally times, on synthetic routings at 32 / 128 / 256 tokens, the CUDA experts
(_kernel_small) under several (NB, W, S, PF) configs and the Marlin path (_experts) under several moe_block_size
values, and prints BENCH lines. Nothing measured is used by kernel_fn. Run at HEAD; output must be byte-identical."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v14_verify_split.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""v14: v13 plus', '''"""v15a (CHECK-ONLY exploration; never submit): v14 + an in-prepare bench of the experts at 32 / 128 / 256 tokens
(CUDA tensor-core configs vs Marlin block sizes). kernel_fn is v14's, unchanged.

v14: v13 plus''')
rep('''    del freqs
    torch.cuda.synchronize()
    return state
''', '''    del freqs
    torch.cuda.synchronize()
    _bench(state, weights)
    return state


BENCH_SMALL = {32: [(1, 8, 4, 2), (1, 8, 2, 2), (1, 4, 4, 2), (1, 16, 4, 2), (1, 8, 8, 2), (2, 8, 4, 2), (2, 8, 2, 2),
                    (2, 4, 4, 2), (1, 8, 4, 4), (4, 8, 4, 2)]}
BENCH_BM = {32: [8, 16, 32], 128: [8, 16, 32, 48], 256: [16, 32, 48, 64]}


def _synthetic_routes(T, gen, dev):
    """Heavy-tailed expert popularity, top-8 of 256, busiest group of 32 rotated to 0-31 (local ids)."""
    scores = torch.rand((T, N_EXPERTS), generator=gen, device=dev)
    z = torch.randn((N_EXPERTS,), generator=gen, device=dev)
    u = torch.rand((N_EXPERTS,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    r = scores.topk(TOP_K, dim=-1)
    ids = r.indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS
    return ids.to(torch.int32), (r.values / r.values.sum(-1, keepdim=True)).float()


def _btime(fn, flush, reps=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    ts = []
    for _ in range(reps):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], out


def _bench(state, weights):
    import time
    t0 = time.time()
    dev = weights["qkv_gate"].device
    flush = torch.empty((256 * 1024 * 1024 // 4,), device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(1234)
    for T in (32, 128, 256):
        ids, tw = _synthetic_routes(T, gen, dev)
        x = torch.randn((T, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
        loc = ids < LOCAL_EXPERTS
        print(f"BENCH T={T} local_routes={int(loc.sum())} experts={int(torch.unique(ids[loc]).numel())} "
              f"max_per_expert={int(torch.bincount(ids[loc].long(), minlength=32).max())}", flush=True)
        ref = None
        for bm in BENCH_BM[T]:
            MOE_BLOCK_M_OVERRIDE[T] = bm
            try:
                t, out = _btime(lambda: _experts(x, tw, ids, state), flush)
                ref = out if ref is None else ref
                print(f"BENCH T={T} marlin block_m={bm} {t:.1f}us", flush=True)
            except Exception as ex:
                print(f"BENCH T={T} marlin block_m={bm} error {type(ex).__name__}: {str(ex)[:160]}", flush=True)
        MOE_BLOCK_M_OVERRIDE.pop(T, None)
        if T == 128 or T == 256:
            MOE_BLOCK_M_OVERRIDE[T] = {128: 16, 256: 32}[T]
        for nb, w, sp, pf in BENCH_SMALL.get(T, []):
            if time.time() - t0 > 240:
                print("BENCH time cap", flush=True)
                return
            data = {"hidden_states": x, "topk_ids": ids,
                    "topk_weights": tw, "config": {"d_expert": D_EXPERT, "top_k": TOP_K, "local_start": 0,
                                                   "local_count": LOCAL_EXPERTS}}
            try:
                t, out = _btime(lambda: _kernel_small(data, state["moe_small"], (nb, (w, sp, pf), (w, sp, pf))), flush)
                err = float((out.float() - ref.float()).norm() / ref.float().norm()) if ref is not None else -1.0
                print(f"BENCH T={T} cuda NB={nb} W={w} S={sp} PF={pf} {t:.1f}us err_vs_marlin={err:.4f}", flush=True)
            except Exception as ex:
                print(f"BENCH T={T} cuda NB={nb} W={w} S={sp} PF={pf} error {type(ex).__name__}: {str(ex)[:160]}",
                      flush=True)
    del flush
    torch.cuda.synchronize()
    print(f"BENCH done in {time.time() - t0:.0f}s", flush=True)
''')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v15a_moe_bench.py"
out.write_text(s)
