"""CPU emulation of gemm_wg's data movement under the documented no-swizzle K-major descriptor model:
operand element (row, k) of a 64x16 (or NTx16) tile sits at start + (row // 8) * SBO + (row % 8) * 16
+ (k // 8) * LBO + (k % 8) * 2 bytes. Checks repack + staging + descriptors + D fragment -> out mapping + split-K."""
import numpy as np, torch

def repack(w):  # the prepare() repack (torch), [N, K] -> flat
    n, k = w.shape
    v = w.contiguous().view(n // 64, 8, 8, k // 64, 4, 2, 8)          # r, rg, row, c, u, kg, e
    return v.permute(0, 3, 4, 5, 1, 2, 6).contiguous().view(-1)        # r, c, u, kg, rg, row, e

def fetch(buf, start, lbo, sbo, rows):  # bytes as uint16 array; returns [rows, 16] values
    out = np.zeros((rows, 16), np.float32)
    for row in range(rows):
        for k in range(16):
            a = start + (row // 8) * sbo + (row % 8) * 16 + (k // 8) * lbo + (k % 8) * 2
            assert a % 2 == 0
            out[row, k] = buf[a // 2]
    return out

def run(M, N, K, S, NT, seed=0, ws=None, cnt=None):
    g = torch.Generator().manual_seed(seed)
    W = torch.randint(-8, 8, (N, K), generator=g).float()
    x = torch.randint(-8, 8, (M, K), generator=g).float()
    wp = repack(W).numpy()                                             # element view (2 bytes each)
    NCH = K // 64 // S
    out = np.full((M, N), np.nan, np.float32)
    rng = np.random.default_rng(seed)
    order = [(r, s) for r in range(N // 64) for s in range(S)]
    rng.shuffle(order)                                                 # CTAs finish in any order
    for r, s in order:
        if True:
            c0 = s * NCH
            src_el = (r * (K // 64) + c0) * 4096                       # 8192 bytes = 4096 elements per chunk
            # staging of x exactly as the kernel loop
            sx = np.zeros(NCH * NT * 64, np.float32)                   # NT*128 bytes per chunk
            for i in range(NT * NCH * 8):
                tok, j = i % NT, i // NT
                c, u, kg = j >> 3, (j >> 1) & 3, j & 1
                dst = c * (NT * 128) + u * (NT * 32) + kg * (NT * 16) + tok * 16
                if tok < M:
                    sx[dst // 2: dst // 2 + 8] = x[tok, c0 * 64 + j * 8: c0 * 64 + j * 8 + 8].numpy()
            D = np.zeros((64, NT), np.float32)
            for c in range(NCH):
                stage = wp[src_el + c * 4096: src_el + (c + 1) * 4096]  # TMA copies this chunk's 8 KB
                for u in range(4):
                    A = fetch(stage, u * 2048, 1024, 128, 64)
                    B = fetch(sx, c * (NT * 128) + u * (NT * 32), NT * 16, 128, NT)   # [NT tokens, 16 k]
                    D += A @ B.T
            # D fragment -> out (as the kernel's epilogue, summed over splits)
            for warp in range(4):
                for lane in range(32):
                    gg, t = lane >> 2, lane & 3
                    for i in range(NT // 2):
                        j, q = i >> 2, i & 3
                        tok = 8 * j + 2 * t + (q & 1)
                        lrow = warp * 16 + gg + (q >> 1) * 8          # row within the 64-row tile
                        if tok < M:
                            ws[(s * 32 + tok) * N + r * 64 + lrow] = D[lrow, tok]   # __stcg(wsp + tok*N + row)
            cnt[r] += 1
            if cnt[r] == S:                                            # last CTA of tile r
                for i in range(16 * M):                                # float4 items: 4 adjacent rows
                    tok, row = i >> 4, r * 64 + (i & 15) * 4
                    for e in range(4):
                        p = tok * N + row + e
                        a = [0.0] * 4
                        k = 0
                        while k + 4 <= S:
                            for q in range(4): a[q] = np.float32(a[q] + ws[p + (k + q) * 32 * N])
                            k += 4
                        while k < S:
                            a[0] = np.float32(a[0] + ws[p + k * 32 * N]); k += 1
                        out[tok, row + e] = (a[0] + a[1]) + (a[2] + a[3])
                cnt[r] = 0
    ref = (x @ W.T).numpy()
    return np.abs(out - ref).max()


if __name__ == "__main__":
    for M, NT in ((8, 8), (5, 8), (16, 16), (11, 16), (32, 32), (19, 32)):
        for N, K, S in ((128, 1024, 4), (192, 2048, 8), (64, 1536, 6), (128, 256, 1 + 0)):
            if S == 1: continue
            ws = np.full(S * 32 * N, 1e30, np.float32)                 # stale garbage: must never be read
            cnt = np.zeros(N // 64, np.int64)
            e1 = run(M, N, K, S, NT, seed=M + K, ws=ws, cnt=cnt)
            e2 = run(M, N, K, S, NT, seed=M + K + 7, ws=ws, cnt=cnt)    # second call reuses ws/cnt unzeroed
            assert e1 == 0 and e2 == 0 and not cnt.any(), (M, NT, N, K, S, e1, e2)
    print("v29 emu: exact, NaN-free, cnt back to 0, ws reuse OK")
