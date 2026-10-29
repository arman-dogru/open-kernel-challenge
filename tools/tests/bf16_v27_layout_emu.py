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

def run(M, N, K, S, NT, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randint(-8, 8, (N, K), generator=g).float()
    x = torch.randint(-8, 8, (M, K), generator=g).float()
    wp = repack(W).numpy()                                             # element view (2 bytes each)
    NCH = K // 64 // S
    out = np.zeros((M, N), np.float32)
    for r in range(N // 64):
        for s in range(S):
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
                            out[tok, r * 64 + lrow] += D[lrow, tok]
    ref = (x @ W.T).numpy()
    return np.abs(out - ref).max()

for M, NT in ((8, 8), (5, 8), (16, 16), (12, 16), (32, 32), (17, 32)):
    print(M, NT, run(M, 128, 512, 2, NT))
