"""(v32: groups of KS CTAs without clusters, RT rowtiles per stage, global-workspace split-K.)
CPU emulation of tools/bf16_v32_stream.cu (no GPU): the prepare()-time repack (_v31_repack, incl. the router's
hi/lo rowtile interleave) and, per launch config, the kernel's byte addressing: the producer's bulk copies (x slice of
rank r into rows of XROW bytes, W stage c of rowtile g = bytes [((rt0+g)*PPR + kp0)*1024 + c*SB, +SB)), the consumer
warps' pair split (i = warp, warp+NW, ... < SPS), A fragments read at ring + i*1024 + lane*16 (+512 for step 1),
B fragments at xs + (8j + lane/4)*XROW + 16*(lane%4) + (c*SPS + i)*64, the m16n8k16 fragment semantics, the per-warp
partial layout ((j*4+c)*32+lane), the fixed-order warp / rank / hi-lo sums and the output index math. Exact vs FP64
on integer data for scaled-down layer shapes with the real plans' K / CS / G structure, m = 1, 5, 8, 13, 16, 32.
Usage: /workspace/okc-cpu-venv/bin/python tools/tests/bf16_v31_emu.py"""
import itertools, sys
import numpy as np
import torch
sys.path.insert(0, "tools")


def repack(w):
    """[n, K] bf16 -> [n/16, K/32, 2 (s), 32 (lane), 8 bf16] (same code as the runs file's _v31_repack)."""
    n, K = w.shape
    t = w.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2)  # rt, h, g, p, t, s, e, q
    return t.permute(0, 3, 5, 2, 4, 6, 1, 7).contiguous()  # rt, p, s, g, t, e, h, q


def router_rows(hi, lo):
    """[256, K] hi, lo -> [512, K]: rowtile 2c = hi rows 16c.., rowtile 2c+1 = lo rows 16c.. (runs file's code)."""
    n, K = hi.shape
    return torch.stack([hi.view(n // 16, 16, K), lo.view(n // 16, 16, K)], 1).reshape(2 * n, K)


lanes = np.arange(32)
gq, tq = lanes // 4, lanes % 4


def frag_mma(a, b):
    """a: [32 lanes, 8 bf16] (regs a0..a3 x 2 halves), b: [32, 4] (b0 halves, b1 halves) -> D regs [32, 4]."""
    A = np.zeros((16, 16))
    A[gq[:, None], 2 * tq[:, None] + np.arange(2)] = a[:, 0:2]
    A[gq[:, None] + 8, 2 * tq[:, None] + np.arange(2)] = a[:, 2:4]
    A[gq[:, None], 2 * tq[:, None] + 8 + np.arange(2)] = a[:, 4:6]
    A[gq[:, None] + 8, 2 * tq[:, None] + 8 + np.arange(2)] = a[:, 6:8]
    B = np.zeros((16, 8))
    B[2 * tq[:, None] + np.arange(2), gq[:, None]] = b[:, 0:2]
    B[2 * tq[:, None] + 8 + np.arange(2), gq[:, None]] = b[:, 2:4]
    D = A @ B
    return np.stack([D[gq, 2 * tq], D[gq, 2 * tq + 1], D[gq + 8, 2 * tq], D[gq + 8, 2 * tq + 1]], 1)


def emulate(wr, x, m, n_out, K, ks_, G, nw, st, sps, hl, rt_):
    nt = (m + 7) // 8
    E = nt * 128
    SB = sps * 1024
    Wb = wr.reshape(-1).float().numpy().astype(np.float64)  # element (2-byte) addressed
    rows = wr.shape[0] * 16
    PPR, Pc = K // 32, K // 32 // ks_
    nst = Pc // sps
    XROW = Pc * 64 + 64
    xv = x.float().numpy().astype(np.float64)
    out = np.full((m, n_out), np.nan)
    ngrp = rows // 16 // G
    ws = np.zeros((ngrp * ks_, G, E))
    for grp in range(ngrp):
        rt0 = grp * G
        for ks in range(ks_):
            cta = grp * ks_ + ks
            kp0 = ks * Pc
            xs = np.full((8 * nt * XROW) // 2, np.nan)
            for c in range(nst):  # x chunk c: rows < m only
                for r in range(m):
                    d = (r * XROW + c * sps * 64) // 2
                    xs[d: d + sps * 32] = xv[r, (kp0 + c * sps) * 32: (kp0 + c * sps) * 32 + sps * 32]
            for g in range(0, G, rt_):
                red = np.zeros((nw, rt_ * E))
                for w in range(nw):
                    acc = np.zeros((rt_, nt, 32, 4))
                    for c in range(nst):
                        # stage: RT bulk copies, rowtile r at ring + r * SB
                        ring = np.concatenate([Wb[(((rt0 + g + r) * PPR + kp0 + c * sps) * 1024) // 2:][: SB // 2]
                                               for r in range(rt_)])
                        for i in range(w, sps, nw):
                            a0 = [ring[(r * SB + i * 1024 + lanes[:, None] * 16) // 2 + np.arange(8)] for r in range(rt_)]
                            a1 = [ring[(r * SB + i * 1024 + 512 + lanes[:, None] * 16) // 2 + np.arange(8)] for r in range(rt_)]
                            for j in range(nt):
                                off = (8 * j + gq) * XROW + 16 * tq + (c * sps + i) * 64
                                b = xs[off[:, None] // 2 + np.arange(8)]
                                for r in range(rt_):
                                    with np.errstate(invalid="ignore"):
                                        acc[r, j] += frag_mma(a0[r], b[:, 0:4]) + frag_mma(a1[r], b[:, 4:8])
                    for r, j, cc in itertools.product(range(rt_), range(nt), range(4)):
                        red[w, r * E + (j * 4 + cc) * 32 + lanes] = acc[r, j, :, cc]
                tot = np.zeros(rt_ * E)
                for w in range(nw):
                    tot = tot + red[w]
                for r in range(rt_):
                    ws[cta, g + r] = tot[r * E:(r + 1) * E]
        GO = G // hl
        for o in range(GO):
            v = np.zeros(E)
            for r in range(ks_):
                for h in range(hl):
                    v = v + ws[grp * ks_ + r, o * hl + h]
            for e in range(E):
                ln, c, j = e & 31, (e >> 5) & 3, e >> 7
                row = (grp * GO + o) * 16 + (ln >> 2) + 8 * (c >> 1)
                col = 8 * j + 2 * (ln & 3) + (c & 1)
                if col < m:
                    out[col, row] = v[e]
    return out


def main():
    torch.manual_seed(0)
    ok = True
    # (name, rows, K, cs, G, hl): real K/CS/Pc structure, fewer clusters
    cases = [("qkv-like", 16 * 9 * 2, 8192, 8, 9, 1), ("qkv-ks16", 16 * 18, 8192, 16, 18, 1), ("o-like", 16 * 8, 1024, 1, 4, 1),
             ("o-ks2", 16 * 16, 1024, 2, 8, 1), ("pooler-like", 16 * 8, 8192, 2, 4, 1), ("unpooler-like", 16 * 16, 4096, 4, 16, 1),
             ("unp_s-like", 16 * 3, 4096, 2, 1, 1), ("router-like", 16 * 4, 8192, 8, 2, 2), ("router-ks16", 16 * 8, 8192, 16, 4, 2)]
    tunes = [(8, 4, 32, 1), (4, 4, 32, 1), (8, 4, 16, 2)]
    for (name, rows, K, cs, G, hl), m in itertools.product(cases, (1, 5, 8, 13, 16, 32)):
        if name in ("qkv-like", "qkv-ks16", "pooler-like") and m not in (1, 13, 32):
            continue
        cs = ks_ = cs
        pc = K // 32 // cs
        for nw, st, sps, rt in tunes:
            if pc % sps or sps % nw or G % rt:
                continue
            if hl == 1:
                w = torch.randint(-8, 9, (rows, K)).to(torch.bfloat16)
                wv, ref_w = w, w.double()
            else:
                hi = torch.randint(-8, 9, (rows // 2, K)).to(torch.bfloat16)
                lo = torch.randint(-3, 4, (rows // 2, K)).to(torch.bfloat16)
                wv, ref_w = router_rows(hi, lo), hi.double() + lo.double()
            x = torch.randint(-4, 5, (m, K)).to(torch.bfloat16)
            ref = (x.double() @ ref_w.T).numpy()
            out = emulate(repack(wv), x, m, ref.shape[1], K, cs, G, nw, st, sps, hl, rt)
            good = np.array_equal(out, ref)
            ok &= good
            print(f"{name} m {m} NW {nw} ST {st} SPS {sps} RT {rt}: {'exact' if good else 'MISMATCH'}", flush=True)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
