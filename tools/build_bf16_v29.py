"""Builds muse_bf16_proj/runs/v29_wgmma_partials.py from v28_wgmma_pipe.py + tools/bf16_v29_wgmma.cu (exact
replacements, each asserted once). Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_bf16_proj/runs/v28_wgmma_pipe.py").read_text()
cu28 = (ROOT / "tools/bf16_v28_wgmma.cu").read_text()
cu29 = (ROOT / "tools/bf16_v29_wgmma.cu").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""v28 (exploration):', '''"""v29 (exploration, offline-verified only): v28 with the split-K FP32 atomics replaced by plain per-split partial
stores ws[s][tok][n] ([S][32][N] FP32) and a last-CTA (counter) reduction that sums s = 0..S-1 in a fixed order
(deterministic; float4 over 4 rows, 4 accumulators), so no zeroing of ws. Routing is per (layer, m) through
WG_ROUTE: the wgmma path is used only where measured (v28 check c20bde96ace6489d) or clearly projected to beat v17.
None qualified (see STATUS.md), so WG_ROUTE is empty and v29 runs v17's routes everywhere; prepare() skips the wgmma
repack / workspaces / NVRTC compile when WG_ROUTE is empty.

v28 (exploration):''')
rep(cu28, cu29)
rep('''WG_MIN_M, WG_MAX_M, WG_ST = 8, 32, 4
''', '''WG_MIN_M, WG_MAX_M, WG_ST = 8, 32, 4
# (WG_SPLIT key, m) pairs routed to the wgmma kernel. Empty: no v28-measured or clearly projected win over v17.
# Closest projected candidates (v28 minus fitted atomic cost plus partial traffic and reduction tail; not clear wins):
# ("pooler", 8) ~20.5 vs v17 20.3 us, ("qkv_gate", 8) ~14.7 vs 14.0 us.
WG_ROUTE = frozenset()


def _wg_key(layer, n):
    return "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
''')
rep('''    S = WG_SPLIT["unpooler_s" if (layer == "unpooler" and n == 1024) else layer]''', '''    S = WG_SPLIT[_wg_key(layer, n)]''')
rep('''            state[name + "_wg"] = _wg_repack(weights[name])
            state[name + "_wgws"] = torch.zeros((WG_MAX_M, n), device=weights[name].device, dtype=torch.float32)
            state[name + "_wgcnt"] = torch.zeros((n // 64,), device=weights[name].device, dtype=torch.int32)
        import cupy as cp
        drv = cp.cuda.driver
        attr = getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8)
        for nt in (8, 16, 32):  # NVRTC-compile now; allow up to 160 KB of dynamic shared memory
            drv.funcSetAttribute(_wg_mod(nt).ptr, attr, 160 * 1024)
''', '''            if WG_ROUTE:
                state[name + "_wg"] = _wg_repack(weights[name])
                # [S][32][N] FP32 partials, fully overwritten by every call before they are read: no zeroing needed
                state[name + "_wgws"] = torch.empty((max(WG_SPLIT.values()) * 32, n), device=weights[name].device,
                                                    dtype=torch.float32)
                state[name + "_wgcnt"] = torch.zeros((n // 64,), device=weights[name].device, dtype=torch.int32)
        if WG_ROUTE:
            import cupy as cp
            drv = cp.cuda.driver
            attr = getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8)
            for nt in (8, 16, 32):  # NVRTC-compile now; allow up to 160 KB of dynamic shared memory
                drv.funcSetAttribute(_wg_mod(nt).ptr, attr, 160 * 1024)
''')
rep('''    if layer != "router" and WG_MIN_M <= m <= WG_MAX_M and x.is_cuda:''',
    '''    if layer != "router" and (_wg_key(layer, n), m) in WG_ROUTE and x.is_cuda:''')

dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_bf16_proj/runs/v29_wgmma_partials.py"
dst.write_text(s)
