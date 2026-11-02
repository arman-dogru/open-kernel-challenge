"""MoE v27 launch configs (shared by the runs files and the offline tests). Config = (GATED, NT, NW, ST, TPS)."""
SMEM_LIMIT = 227 * 1024 - 2048  # dynamic bytes (the kernel has ~1.6 KB static)
SMEM_MIN = 116 * 1024           # one CTA per SM


def smem(cfg, K):
    gated, nt, nw, st, tps = cfg
    halves = 2 if gated else 1
    return st * halves * tps * 576 + 8 * nt * (K + 8) * 2 + nw * halves * nt * 128 * 4 + 2 * st * 8


def valid(cfg, K=4096):
    gated, nt, nw, st, tps = cfg
    return (K // 64) % tps == 0 and tps % nw == 0 and smem(cfg, K) <= SMEM_LIMIT


# per GEMM: candidate (NW, ST, TPS) tunes
TUNES = [(8, 4, 16), (8, 3, 32), (8, 4, 32), (16, 3, 32), (16, 4, 32), (8, 2, 64), (16, 2, 64), (16, 3, 64)]
ALL_CONFIGS = [(g, nt, nw, st, tps) for g in (1, 0) for nt in (1, 2) for (nw, st, tps) in TUNES
               if valid((g, nt, nw, st, tps))]
