"""bf16 v30 kernel configs (shared by the build script, the runs files' host code and the offline tests).
A config is (RT, WK, RG, CS, GP): RT rowtiles (16 rows) per warp, WK warps along k per CTA, RG rowtile groups per CTA,
CS CTAs per cluster along k, GP pairs (32 k) of A fragments in flight per warp. NT (token tiles of 8) = ceil(m / 8)."""
CONFIGS = [(1, 8, 1, 1, 8), (1, 16, 1, 1, 8), (1, 8, 1, 1, 16), (2, 8, 1, 1, 8), (1, 4, 2, 1, 8), (1, 8, 1, 2, 8),
           (1, 8, 1, 4, 4), (1, 8, 1, 2, 4), (2, 4, 2, 1, 8), (1, 4, 4, 1, 4), (1, 8, 2, 1, 4), (2, 8, 1, 2, 8),
           (1, 4, 2, 2, 8), (2, 4, 1, 2, 8)]
NTS = (1, 2, 4)
ALL_INSTANCES = [(rt, nt, wk, rg, cs, gp) for (rt, wk, rg, cs, gp) in CONFIGS for nt in NTS]
# layer -> (n rows used, K)
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "unpooler_s": (1024, 4096)}


def name_expr(inst):
    rt, nt, wk, rg, cs, gp = inst
    return f"bf16_stream<{rt}, {nt}, {wk}, {rg}, {cs}, {gp}>"


def valid(cfg, n, K, nt):
    rt, wk, rg, cs, gp = cfg
    rows = n // 16
    if n % 16 or rows % (rg * rt) or K % 32:
        return False
    kp = K // 32
    if kp % (wk * cs):
        return False
    pw = kp // (wk * cs)
    if pw < gp or pw % gp:
        return False
    return wk * rg * 32 <= 512 and smem(cfg, nt) <= 96 * 1024


def smem(cfg, nt):
    rt, wk, rg, cs, gp = cfg
    return wk * rg * rt * nt * 128 * 4


def grid(cfg, n, cap=264):
    """CTAs: one cluster per unit up to cap CTAs (persistent: clusters loop over units)."""
    rt, wk, rg, cs, gp = cfg
    units = (n // 16) // (rg * rt)
    return min(units, max(1, cap // cs)) * cs, units
