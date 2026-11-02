"""bf16 v31 configuration (shared by the runs files, the NVRTC compile test and the CPU emulation).
Instance (template args) = (NT, CS, NW, ST, SPS, HL, F32). A launch also picks G (rowtiles per cluster).
A layer's plan = (CS, G): grid = rows/16/G clusters of CS CTAs (<= 132 CTAs, one per SM)."""
SMS = 132
SMEM_MAX = 227 * 1024
SMEM_MIN = 116 * 1024  # request at least this much so two CTAs never share an SM
# key -> (rows in the repacked weight, K, HL, F32, n_out); router rows are [hi 16 rows, lo 16 rows] per output rowtile
LAYERS = {"qkv_gate": (2304, 8192, 1, False, 2304), "o": (8192, 1024, 1, False, 8192),
          "pooler": (4096, 8192, 1, False, 4096), "unpooler": (8192, 4096, 1, False, 8192),
          "unpooler_s": (1024, 4096, 1, False, 1024), "router": (512, 8192, 2, True, 256)}
PLANS = {"qkv_gate": [(8, 9)], "o": [(1, 4), (2, 8), (4, 16)], "pooler": [(8, 16), (4, 8), (2, 4)],
         "unpooler": [(4, 16), (8, 32), (2, 8)], "unpooler_s": [(2, 1), (4, 2), (8, 4)], "router": [(8, 2)]}
TUNES = [(4, 4, 32), (8, 4, 32), (4, 8, 16), (4, 3, 32), (4, 6, 16), (8, 3, 32)]  # (NW consumer warps, ST stages, SPS pairs (KB) per stage)
NTS = (1, 2, 4)


def smem(inst, K, G):
    nt, cs, nw, st, sps, hl, f32 = inst
    E = nt * 128
    pc = K // 32 // cs
    seg = cs > 1 or hl > 1
    return st * sps * 1024 + 8 * nt * (pc * 64 + 64) + 2 * nw * E * 4 + (G * E * 4 if seg else 0) + (2 * st + 1) * 8


def valid(key, nt, plan, tune):
    rows, K, hl, f32, _ = LAYERS[key]
    cs, G = plan
    nw, st, sps = tune
    pc = K // 32 // cs
    nrt = rows // 16
    return (K // 32) % cs == 0 and pc % sps == 0 and pc % 2 == 0 and sps % nw == 0 and nrt % G == 0 \
        and G % hl == 0 and nrt // G * cs <= SMS and smem((nt, cs, nw, st, sps, hl, f32), K, G) <= SMEM_MAX


def instance(key, nt, plan, tune):
    _, _, hl, f32, _ = LAYERS[key]
    return (nt, plan[0], tune[0], tune[1], tune[2], hl, f32)


def name_expr(inst):
    nt, cs, nw, st, sps, hl, f32 = inst
    return f"bf16_v31<{nt},{cs},{nw},{st},{sps},{hl},{'true' if f32 else 'false'}>"


def candidates(key, nt):
    return [(p, t) for p in PLANS[key] for t in TUNES if valid(key, nt, p, t)]


ALL_INSTANCES = sorted({instance(k, nt, p, t) for k in LAYERS for nt in NTS for (p, t) in candidates(k, nt)})


def launch(key, plan, tune, nt):
    """(grid CTAs, threads, dynamic smem bytes)."""
    rows, K, hl, f32, _ = LAYERS[key]
    cs, G = plan
    return rows // 16 // G * cs, (tune[0] + 1) * 32, max(SMEM_MIN, smem(instance(key, nt, plan, tune), K, G))
