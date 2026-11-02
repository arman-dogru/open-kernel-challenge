"""CPU emulation of tools/bf16_v30_stream.cu (no GPU): the prepare()-time W repack (runs file's _v30_repack) plus,
per config, every warp's loads exactly as the kernel addresses them (W fragment words from the repacked buffer, x
words at k = 32 p + 8 t + 4 s + 0..3), the m16n8k16 MMA by its PTX fragment layouts (A: a0a1 row g k 2t..+1,
a2a3 row g+8, a4a5 row g k 2t+8..+9, a6a7 row g+8; B: b0b1 k 2t..+1 col g, b2b3 k 2t+8..+9; C: c0c1 row g cols 2t..+1,
c2c3 row g+8), the shared-memory element order of the in-CTA reduction, the cluster sum and the output index math.
Checks exact equality with an FP64 x . W^T on integer-valued data (all sums exact in FP32), for every layer shape
(n scaled down) and config, m = 1, 5, 8, 13, 16, 32. Usage: /workspace/okc-cpu-venv/bin/python tools/tests/bf16_v30_emu.py"""
import itertools, sys
import numpy as np
import torch
sys.path.insert(0, "tools")
from bf16_v30_cfg import CONFIGS, valid


def repack(w):
    """[n, K] bf16 -> [n/16, K/32, 2 (s), 32 (lane), 8 bf16] (same code as the runs file's _v30_repack)."""
    n, K = w.shape
    t = w.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2)  # rt, h, g, p, t, s, e, q
    return t.permute(0, 3, 5, 2, 4, 6, 1, 7).contiguous()  # rt, p, s, g, t, e, h, q


def emulate(wr, x, m, n, K, cfg):
    rt_, wk, rg, cs, gp = cfg
    nt = (m + 7) // 8
    W = wr.view(n // 16, K // 32, 2, 32, 4, 2).float().numpy().astype(np.float64)  # words: [.., lane, reg, half]
    X = np.zeros((nt * 8, K))
    X[:m] = x.float().numpy().astype(np.float64)
    units = (n // 16) // (rg * rt_)
    kp = K // 32
    pw = kp // (wk * cs)
    out = np.full((m, n), np.nan)
    lanes = np.arange(32)
    g, t = lanes // 4, lanes % 4
    for u in range(units):
        E = rg * rt_ * nt * 128
        red = np.zeros((cs, wk, E))
        for rank, r_g, ks in itertools.product(range(cs), range(rg), range(wk)):
            p0 = (rank * wk + ks) * pw
            acc = np.zeros((rt_, nt, 32, 4))
            for r in range(rt_):
                rt = (u * rg + r_g) * rt_ + r
                for p in range(p0, p0 + pw):
                    for s in range(2):
                        a = W[rt, p, s]  # [lane, 4 regs, 2 halves]
                        # build the per-lane A fragment as a 16x16 matrix in MMA coordinates
                        A = np.zeros((16, 16))
                        A[g[:, None], 2 * t[:, None] + np.arange(2)] = a[:, 0]
                        A[g[:, None] + 8, 2 * t[:, None] + np.arange(2)] = a[:, 1]
                        A[g[:, None], 2 * t[:, None] + 8 + np.arange(2)] = a[:, 2]
                        A[g[:, None] + 8, 2 * t[:, None] + 8 + np.arange(2)] = a[:, 3]
                        kb = 32 * p + 8 * t + 4 * s
                        for j in range(nt):
                            B = np.zeros((16, 8))
                            xr = X[8 * j + g]  # [lane, K]
                            B[2 * t[:, None] + np.arange(2), g[:, None]] = xr[lanes[:, None], kb[:, None] + np.arange(2)]
                            B[2 * t[:, None] + 8 + np.arange(2), g[:, None]] = xr[lanes[:, None], kb[:, None] + 2 + np.arange(2)]
                            D = A @ B  # [16 rows, 8 cols]
                            acc[r, j] += np.stack([D[g, 2 * t], D[g, 2 * t + 1], D[g + 8, 2 * t], D[g + 8, 2 * t + 1]], 1)
            base = r_g * rt_ * nt * 128
            for r, j, c, l in itertools.product(range(rt_), range(nt), range(4), range(32)):
                red[rank, ks, base + ((r * nt + j) * 4 + c) * 32 + l] = acc[r, j, l, c]
        tot = red.sum(1).sum(0)
        for e in range(E):
            ln, c, rj = e & 31, (e >> 5) & 3, e >> 7
            j, R = rj % nt, rj // nt
            row = (u * rg * rt_ + R) * 16 + (ln >> 2) + 8 * (c >> 1)
            col = 8 * j + 2 * (ln & 3) + (c & 1)
            if col < m and row < n:
                out[col, row] = tot[e]
    return out


def main():
    torch.manual_seed(0)
    ok = True
    # scaled-down shapes with the real K where cheap; n small so the emulation is quick
    shapes = [("qkv-like", 64, 2048), ("o-like", 128, 1024), ("unp_s-like", 32, 4096)]
    for (name, n, K), m in itertools.product(shapes, (1, 5, 8, 13, 16, 32)):
        w = torch.randint(-8, 9, (n, K)).to(torch.bfloat16)
        x = torch.randint(-4, 5, (m, K)).to(torch.bfloat16)
        ref = (x.double() @ w.double().T).numpy()
        wr = repack(w)
        nt = (m + 7) // 8
        tried = 0
        for cfg in CONFIGS:
            if not valid(cfg, n, K, nt):
                continue
            tried += 1
            out = emulate(wr, x, m, n, K, cfg)
            good = np.array_equal(out, ref)
            ok &= good
            if not good:
                print("MISMATCH", name, m, cfg, np.nanmax(np.abs(out - ref)))
        print(f"{name} n {n} K {K} m {m}: {tried} configs exact", flush=True)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
