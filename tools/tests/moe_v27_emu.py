"""CPU emulation of tools/moe_v27_stream.cu (no GPU), FASTSCALE path, shrunken dims (H = I = 512), small grids.
Emulates what the kernel addresses and in which order: the per-CTA segment list from cnt[] (chunks of TC routes),
the equal split of the stage sequence q = (segment * RP + p) * SPU + kc over P CTAs, each stage's bytes at
w + t0 * 512 / sf + t0 * 64 with t0 = ((ex * RBT + h * RP + p) * KB + kc * TPS) (h = gate / up half), the consumer
warps' tile split (i = warp, warp + NW, ...), the A fragments decoded from the stage bytes exactly as mma_tile does
(rotl / AND masks, PRMT + IMAD scale splat, bf16 products), B fragments from the staged activation rows, the per-warp
partial order, the hand-over of a unit shared by two CTAs (partial of the later CTA added by the earlier one), and the
epilogue (gs * 128, silu(gate) * up rounded to BF16 for GEMM 1, FP32 y for GEMM 2). Compares y against an FP64
dequantize-and-matmul reference of the same routes (with the same BF16 rounding of silu(gate) * up).
Usage: /workspace/okc-cpu-venv/bin/python tools/tests/moe_v27_emu.py"""
import itertools, os, sys
import numpy as np
import torch
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "muse_moe_w4a16")
sys.path.insert(0, d)
sys.path.insert(0, d + "/runs")
import reference as R
import nvfp4
import v26e_nb2_evict as V  # _repack_bank (the v26 A-fragment repack the kernel streams)

lanes = np.arange(32)
HANDOVERS = [0]
gq, tq = lanes // 4, lanes % 4


def bf(bits16):
    return (bits16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def rotl(w, n):
    return ((w << np.uint32(n)) | (w >> np.uint32(32 - n))) & np.uint32(0xFFFFFFFF)


def splat(sw, k):
    b = (sw >> np.uint32(8 * k)) & np.uint32(0xFF)
    return bf(((b << np.uint32(4)) + np.uint32(0x7780)) & np.uint32(0xFFFF))  # one half (both halves equal)


def halves(a):
    return bf(a & np.uint32(0xFFFF)), bf(a >> np.uint32(16))


def tile_A(code_bytes, sc_bytes):
    """-> [4 k-steps][32 lanes][8 values] (a0 lo/hi, a1, a2, a3) as mma_tile builds them (FASTSCALE)."""
    q = np.frombuffer(code_bytes, dtype=np.uint32).reshape(32, 4)
    s = np.frombuffer(sc_bytes, dtype=np.uint32).reshape(8, 2)[gq]  # lane's uint2 at g * 8
    out = np.zeros((4, 32, 8))
    for st in range(4):
        wq = q[:, st]
        sw = s[:, 0] if st < 2 else s[:, 1]
        slo, shi = splat(sw, 2 if st & 1 else 0), splat(sw, 3 if st & 1 else 1)
        regs = [(rotl(wq, 6) & np.uint32(0x81C081C0), slo), (rotl(wq, 2) & np.uint32(0x81C081C0), shi),
                (rotl(wq, 28) & np.uint32(0x81C081C0), slo),
                ((wq & np.uint32(0x81808180)) | (rotl(wq, 24) & np.uint32(0x00400040)), shi)]
        for i, (a, sc) in enumerate(regs):
            lo, hi = halves(a)
            out[st, :, 2 * i] = lo * sc
            out[st, :, 2 * i + 1] = hi * sc
    return out


def frag_mma(a, b0, b1):
    A = np.zeros((16, 16))
    A[gq[:, None], 2 * tq[:, None] + np.arange(2)] = a[:, 0:2]
    A[gq[:, None] + 8, 2 * tq[:, None] + np.arange(2)] = a[:, 2:4]
    A[gq[:, None], 2 * tq[:, None] + 8 + np.arange(2)] = a[:, 4:6]
    A[gq[:, None] + 8, 2 * tq[:, None] + 8 + np.arange(2)] = a[:, 6:8]
    B = np.zeros((16, 8))
    B[2 * tq[:, None] + np.arange(2), gq[:, None]] = b0
    B[2 * tq[:, None] + 8 + np.arange(2), gq[:, None]] = b1
    with np.errstate(invalid="ignore"):
        D = A @ B
    return np.stack([D[gq, 2 * tq], D[gq, 2 * tq + 1], D[gq + 8, 2 * tq], D[gq + 8, 2 * tq + 1]], 1)


def emulate(gated, xin, xrow_div, w, sf, gs, cnt, lst, maxm, local_start, N, K, NT, NW, TPS, P):
    TC, E, H2 = 8 * NT, NT * 128, 2 if gated else 1
    RP, KB = N // 16, K // 64
    SPU, RBT = KB // TPS, H2 * N // 16
    wb, sb = w.numpy().tobytes(), sf.numpy().tobytes()
    segs = [(e, i * TC, min(TC, int(cnt[e]) - i * TC)) for e in range(32) for i in range((int(cnt[e]) + TC - 1) // TC)]
    U = len(segs) * RP * SPU
    nroutes = lst.shape[0] // 1
    out = {}
    ws = np.zeros((P, H2 * E))
    finals = []  # (cta, u, partial, shared_next)
    for b in range(P):
        q0, q1 = U * b // P, U * (b + 1) // P
        cur = None
        xs = None
        acc = np.zeros((NW, H2, NT, 32, 4))
        for q in range(q0, q1):
            kc, u = q % SPU, q // SPU
            p, sg = u % RP, u // RP
            ex, i0, n = segs[sg]
            if sg != cur:
                xs = np.full((TC, K), np.nan)
                for m in range(n):
                    r = int(lst[ex * maxm + i0 + m])
                    xs[m] = xin[r // xrow_div]
                cur = sg
            for h in range(H2):
                t0 = ((ex * RBT + h * RP + p) * KB + kc * TPS)
                cb = wb[t0 * 512: t0 * 512 + TPS * 512]
                sbb = sb[t0 * 64: t0 * 64 + TPS * 64]
                for wi in range(NW):
                    for i in range(wi, TPS, NW):
                        kb = kc * TPS + i
                        A = tile_A(cb[i * 512:(i + 1) * 512], sbb[i * 64:(i + 1) * 64])
                        for nb, st in itertools.product(range(NT), range(4)):
                            xr = xs[nb * 8 + gq]
                            k = kb * 64 + st * 16 + 2 * tq
                            b0 = np.stack([xr[lanes, k], xr[lanes, k + 1]], 1)
                            b1 = np.stack([xr[lanes, k + 8], xr[lanes, k + 9]], 1)
                            acc[wi, h, nb] += frag_mma(A[st], b0, b1)
            if not (kc == SPU - 1 or q == q1 - 1):
                continue
            red = np.zeros((NW, H2, E))
            for wi, h, j, c in itertools.product(range(NW), range(H2), range(NT), range(4)):
                red[wi, h, (j * 4 + c) * 32 + lanes] = acc[wi, h, j, :, c]
            acc[:] = 0
            tot = np.zeros((H2, E))
            for wi in range(NW):
                tot = tot + red[wi]
            ustart, uend = u * SPU, u * SPU + SPU
            if ustart < q0:
                ws[b] = tot.reshape(-1)
                HANDOVERS[0] += 1
                continue
            finals.append((b, u, tot, uend > q1))
    for b, u, tot, shared_next in finals:
        if shared_next:
            tot = tot + ws[b + 1].reshape(H2, E)
        p, sg = u % RP, u // RP
        ex, i0, n = segs[sg]
        gsc = float(gs[local_start + ex]) * 128.0
        v = tot * gsc
        for e in range(E):
            ln, c, j = e & 31, (e >> 5) & 3, e >> 7
            row = p * 16 + (ln >> 2) + 8 * (c >> 1)
            col = 8 * j + 2 * (ln & 3) + (c & 1)
            if col < n:
                r = int(lst[ex * maxm + i0 + col])
                if gated:
                    gv, uv = v[0, e], v[1, e]
                    out[(r, row)] = gv / (1.0 + np.exp(-gv)) * uv
                else:
                    out[(r, row)] = v[0, e]
    return out


def main():
    R.D_HIDDEN = R.D_EXPERT = 512
    Hd = Id = 512
    W = R.generate_weights(3, device="cpu")
    start, count = 0, 32
    w13, w13_sf = V._repack_bank(W["w13"], W["w13_sf"], start, count)
    w2, w2_sf = V._repack_bank(W["w2"], W["w2_sf"], start, count)
    sfb = W["w13_sf"][:32].contiguous().view(torch.uint8)
    assert bool(((sfb >= 8) & (sfb <= 0x7E)).all()), "FASTSCALE needs normal scales"
    ok = True
    for (bs, seed, NT, NW, TPS, P) in [(1, 1, 1, 8, 4, 7), (8, 2, 1, 4, 4, 13), (16, 3, 2, 4, 4, 11), (32, 4, 2, 4, 4, 9),
                                       (32, 5, 1, 4, 4, 17)]:
        HANDOVERS[0] = 0
        data = R.generate_input(bs=bs, sharding="ep8", seed=seed, dist="normal", device="cpu")
        x = data["hidden_states"].float().numpy().astype(np.float64)
        ids = data["topk_ids"].reshape(-1).numpy()
        NR, maxm = ids.size, ids.size
        cnt = np.zeros(32, dtype=np.int64)
        lst = np.full(32 * maxm, -1, dtype=np.int64)
        for r, e in enumerate(ids):  # route_prep's lists (any order within an expert: the kernel is order-agnostic)
            if start <= e < start + count:
                lst[(e - start) * maxm + cnt[e - start]] = r
                cnt[e - start] += 1
        h = emulate(1, x, 8, w13, w13_sf, W["w13_gs"], cnt, lst, maxm, start, Id, Hd, NT, NW, TPS, P)
        inter = np.zeros((NR, Id))
        for (r, row), v in h.items():
            inter[r, row] = torch.tensor(v, dtype=torch.float32).to(torch.bfloat16).float().item()
        y = emulate(0, inter, 1, w2, w2_sf, W["w2_gs"], cnt, lst, maxm, start, Hd, Id, NT, NW, TPS, P)
        # reference: same routes, FP64 dequantized weights, BF16 rounding of silu(gate) * up
        err, nchk = 0.0, 0
        for e in range(32):
            rs = [int(r) for r in lst[e * maxm: e * maxm + cnt[e]]]
            if not rs:
                continue
            W13 = nvfp4.dequantize(W["w13"][e], W["w13_sf"][e], W["w13_gs"][e]).double().numpy()
            W2 = nvfp4.dequantize(W["w2"][e], W["w2_sf"][e], W["w2_gs"][e]).double().numpy()
            xr = x[[r // 8 for r in rs]]
            hh = xr @ W13.T
            it = hh[:, :Id] / (1 + np.exp(-hh[:, :Id])) * hh[:, Id:]
            it = torch.tensor(it).float().to(torch.bfloat16).double().numpy()
            yr = it @ W2.T
            got = np.array([[y[(r, c)] for c in range(Hd)] for r in rs])
            err = max(err, float(np.abs(got - yr).max() / np.abs(yr).max()))
            nchk += len(rs)
        missing = sum(1 for r in range(NR) if ids[r] < 32 for c in range(Hd) if (r, c) not in y)
        good = err < 2e-2 and missing == 0 and nchk > 0 and HANDOVERS[0] > 0
        
        ok &= good
        print(f"bs {bs} NT {NT} NW {NW} TPS {TPS} P {P}: routes {nchk}, max rel err {err:.2e}, missing {missing}, hand-overs {HANDOVERS[0]} -> {'ok' if good else 'BAD'}", flush=True)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
