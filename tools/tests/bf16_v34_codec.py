"""CPU test of the bf16 v34 lossless 12-bit weight codec: _v34_pack (taken from the built runs file) + a bit-exact
Python model of the kernel's dec4 decode and esc_fix, against the v33 A-fragment repack of the same weights.
Checks: every in-window weight decodes to its exact BF16 bits in v33's fragment position; every escape's
decoded + ecorr is exactly the weight; a GEMV through decode + fix-up equals x . W^T in FP64.
Usage: /workspace/okc-cpu-venv/bin/python tools/tests/bf16_v34_codec.py [runs file]"""
import ast, math, sys
import numpy as np
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "muse_bf16_proj/runs/v34d_bench.py"
src = open(path).read()
tree = ast.parse(src)
ns = {"torch": torch}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_v34_pack", "_v34_repack"):
        exec(compile(ast.Module([node], []), path, "exec"), ns)
pack, repack = ns["_v34_pack"], ns["_v34_repack"]


def dec4(aw0, aw1, nw, cc):
    M = 0xFFFFFFFF
    y0 = ((nw & 0x000F000F) * 128 + cc) & M
    y1 = ((nw & 0x00F000F0) * 8 + cc) & M
    y2 = (((nw & 0x0F000F00) * 0x80000000 >> 32) + cc) & M
    y3 = (((nw & 0xF000F000) * 0x08000000 >> 32) + cc) & M

    def bp(a, sel):  # __byte_perm(a, 0, sel)
        bs = [(a >> (8 * i)) & 0xFF for i in range(4)] + [0, 0, 0, 0]
        return sum(bs[(sel >> (4 * i)) & 7] << (8 * i) for i in range(4))
    return [(bp(aw0, 0x1100) & 0x807F807F) | y0, (bp(aw0, 0x3322) & 0x807F807F) | y1,
            (bp(aw1, 0x1100) & 0x807F807F) | y2, (bp(aw1, 0x3322) & 0x807F807F) | y3]


bad = 0
for n, K, seed in ((64, 256, 1), (32, 512, 2), (48, 1024, 3)):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn((n, K), generator=g) / math.sqrt(K))
    flat = w.view(-1)
    idx = torch.randperm(flat.numel(), generator=g)[:40]
    flat[idx[:10]] = 0.0
    flat[idx[10:20]] *= 1e-6
    flat[idx[20:30]] *= 3e3
    flat[idx[30:]] = -flat[idx[30:]] * 1e-30
    w = w.to(torch.bfloat16)
    pk, cc, eoff, emeta, ecorr = pack(w)
    ref = repack(w).view(torch.int16).numpy().astype(np.int64) & 0xFFFF  # [rt, p, s, g, t, e, h, q]
    ref = ref.reshape(n // 16, K // 32, 2, 32, 8)
    pk = pk.numpy()
    nesc = int(eoff[-1])
    dec = np.zeros_like(ref)
    for rt in range(n // 16):
        for p in range(K // 32):
            blk = pk[rt, p]
            for lane in range(32):
                ab = blk[lane * 16: lane * 16 + 16].view(np.uint32)
                nb = blk[512 + lane * 8: 512 + lane * 8 + 8].view(np.uint32)
                for s in range(2):
                    regs = dec4(int(ab[2 * s]), int(ab[2 * s + 1]), int(nb[s]), cc)
                    for j, r in enumerate(regs):
                        dec[rt, p, s, lane, 2 * j] = r & 0xFFFF
                        dec[rt, p, s, lane, 2 * j + 1] = r >> 16
    mism = dec != ref
    # map decoded fragment back to [n, K] for the GEMV / escape check
    t = torch.from_numpy(dec.astype(np.int32)).to(torch.int16).view(torch.bfloat16)  # [rt, p, s, lane, 8]
    t = t.reshape(n // 16, K // 32, 2, 8, 4, 2, 2, 2)  # rt, p, s, g, t, e, h, q
    wd = t.permute(0, 6, 3, 1, 4, 2, 5, 7).reshape(n, K)  # inverse of (0,3,5,2,4,6,1,7)
    fixed = wd.float().clone()
    for rt in range(n // 16):
        for e in range(int(eoff[rt]), int(eoff[rt + 1])):
            mt = int(emeta[e])
            assert float(fixed[rt * 16 + (mt >> 16), mt & 0xFFFF]) == 2.0 ** (((cc >> 7) & 0xFF) - 127)
            fixed[rt * 16 + (mt >> 16), mt & 0xFFFF] = float(ecorr[e])
    exact = torch.equal(fixed, w.float())
    x = torch.randn((5, K), generator=g).to(torch.bfloat16)
    y = x.double() @ wd.double().t()
    for rt in range(n // 16):
        for e in range(int(eoff[rt]), int(eoff[rt + 1])):
            mt = int(emeta[e])
            y[:, rt * 16 + (mt >> 16)] += (float(ecorr[e]) - 2.0 ** (((cc >> 7) & 0xFF) - 127)) * x[:, mt & 0xFFFF].double()
    yref = x.double() @ w.double().t()
    gerr = float((y - yref).abs().max() / yref.abs().max())
    # probe rows in the kernel's FP32 order: v = MMA sum (one nonzero product), then (v - dmin x) + w x per escape
    pexact = True
    dmin = np.float32(2.0 ** (((cc >> 7) & 0xFF) - 127))
    for rt in range(n // 16):
        for e in range(int(eoff[rt]), int(eoff[rt + 1])):
            mt = int(emeta[e])
            xk = np.float32(2.0 ** 7)
            v = np.float32(dmin * xk)
            v = np.float32(np.float32(v - np.float32(dmin * xk)) + np.float32(np.float32(ecorr[e]) * xk))
            pexact &= float(v) == float(w[rt * 16 + (mt >> 16), mt & 0xFFFF].float()) * 128.0
    exact = exact and pexact
    ok = int(mism.sum()) == nesc and exact and gerr < 1e-12 and nesc >= 10
    print(f"n {n} K {K}: lo {(cc >> 7) & 0xFF}, escapes {nesc}, fragment mismatches {int(mism.sum())} (= escapes),"
          f" escapes decode to dmin, fixed weights exact {exact}, GEMV rel err {gerr:.1e} -> {'ok' if ok else 'BAD'}")
    bad += not ok
print("PASS" if not bad else "FAIL")
sys.exit(bad)
