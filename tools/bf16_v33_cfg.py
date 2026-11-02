"""bf16 v33 configuration (shared by the runs files, the NVRTC compile test and the CPU emulation).
Instance (template args) = (NT, RT, NW, ST, SPS, HL, F32). A launch also picks plan = (KS, G): grid = rows/16/G
groups of KS CTAs (<= 132 CTAs, one per SM); CTA ks of a group owns k-slice ks of the group's G rowtiles."""
SMS = 132
SMEM_MAX = 226 * 1024  # dynamic (plus a 4-byte static)
SMEM_MIN = 116 * 1024  # request at least this much so two CTAs never share an SM
# key -> (rows in the repacked weight, K, HL, F32, n_out); router rows are [hi 16 rows, lo 16 rows] per output rowtile
LAYERS = {"qkv_gate": (2304, 8192, 1, False, 2304), "o": (8192, 1024, 1, False, 8192),
          "pooler": (4096, 8192, 1, False, 4096), "unpooler": (8192, 4096, 1, False, 8192),
          "unpooler_s": (1024, 4096, 1, False, 1024), "router": (512, 8192, 2, True, 256)}
PLANS = {"qkv_gate": [(8, 9), (16, 18)], "o": [(1, 4), (2, 8), (4, 16)], "pooler": [(2, 4), (4, 8), (8, 16)],
         "unpooler": [(1, 4), (2, 8), (4, 16)], "unpooler_s": [(2, 1), (4, 2), (8, 4)], "router": [(8, 2), (16, 4)]}
# (NW consumer warps, ST stages, SPS pairs (KB) per rowtile per stage, RT rowtiles per stage)
TUNES = [(8, 3, 32, 1), (8, 4, 32, 1), (8, 6, 16, 1), (8, 3, 16, 2), (8, 4, 16, 2), (8, 2, 32, 2), (8, 3, 16, 3),
         (8, 4, 8, 3), (8, 3, 8, 4), (8, 4, 8, 4), (8, 2, 16, 4), (4, 4, 16, 2), (4, 3, 8, 4)]
NTS = (1, 2, 4)


def smem(inst, K, KS):
    nt, rt, nw, st, sps, hl, f32 = inst
    E = nt * 128
    pc = K // 32 // KS
    return st * rt * sps * 1024 + 8 * nt * (pc * 64 + 64) + nw * rt * E * 4 + (2 * st + 16) * 8


def instance(key, nt, plan, tune):
    _, _, hl, f32, _ = LAYERS[key]
    nw, st, sps, rt = tune
    return (nt, rt, nw, st, sps, hl, f32)


def valid(key, nt, plan, tune):
    rows, K, hl, f32, _ = LAYERS[key]
    ks, G = plan
    nw, st, sps, rt = tune
    pc = K // 32 // ks
    nrt = rows // 16
    return (K // 32) % ks == 0 and pc % sps == 0 and pc // sps <= 16 and pc % 2 == 0 and sps % nw == 0 \
        and nrt % G == 0 and G % rt == 0 and G % hl == 0 and nrt // G * ks <= SMS \
        and smem(instance(key, nt, plan, tune), K, ks) <= SMEM_MAX


def name_expr(inst):
    nt, rt, nw, st, sps, hl, f32 = inst
    return f"bf16_v33<{nt},{rt},{nw},{st},{sps},{hl},{'true' if f32 else 'false'}>"


def candidates(key, nt):
    return [(p, t) for p in PLANS[key] for t in TUNES if valid(key, nt, p, t)]


ALL_INSTANCES = sorted({instance(k, nt, p, t) for k in LAYERS for nt in NTS for (p, t) in candidates(k, nt)})


def launch(key, plan, tune, nt):
    """(grid CTAs, threads, dynamic smem bytes, workspace floats)."""
    rows, K, hl, f32, _ = LAYERS[key]
    ks, G = plan
    grid = rows // 16 // G * ks
    return grid, (tune[0] + 1) * 32, max(SMEM_MIN, smem(instance(key, nt, plan, tune), K, ks)), grid * G * nt * 128
