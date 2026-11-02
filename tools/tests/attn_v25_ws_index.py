# Offline checks for attn_v25_ws.cu index math vs v24 (attn_v25_fv.cu default build).
KROW, BN, VTH = 128, 64, 9216
# 1) loader placement: v24 thread tid -> key kl = tid >> 2, chunks c0, c0+1 (c0 = (tid & 3) * 2) at kl*KROW + ((c ^ (kl&7)) << 4),
#    source byte row*KROW + c*16.  WS producer lane, instr j -> key pk + 4j, chunk pc.
v24 = {}
for tid in range(256):
    kl, c0 = tid >> 2, (tid & 3) * 2
    for c in (c0, c0 + 1):
        v24[(kl, c)] = kl * KROW + ((c ^ (kl & 7)) << 4)
ws = {}
for lane in range(32):
    pk, pc = lane >> 3, lane & 7
    for j in range(16):
        key = pk + 4 * j
        assert (key, pc) not in ws
        ws[(key, pc)] = key * KROW + ((pc ^ (key & 7)) << 4)
        # shuffle source lane / register: key & 31 from ra (j < 8) or rb
        assert (key < 32) == (j < 8) and (key & 31) == (key if key < 32 else key - 32)
assert ws == v24 and len(ws) == 64 * 8, "loader placement differs"
print("loader placement: identical to v24 (512 chunks per tile, K and V)")
# 2) transpose task map: v24 (warp 0..7, rep) vs WS (h, wi, rep) with pw
def task(warp, lane, rep):
    combo = warp + 8 * rep
    dg = (lane & 7) + 8 * (combo & 3); sg = (lane >> 3) + 4 * (combo >> 2)
    blk, sl = sg >> 3, sg & 7; tt, A = sl & 3, sl >> 2
    k0 = blk * 32 + 16 * A + 2 * tt
    trl = k0 * KROW + ((((dg >> 2) ^ (k0 & 7)) * 16) + (dg & 3) * 4)
    trl1 = (k0 + 1) * KROW + ((((dg >> 2) ^ ((k0 + 1) & 7)) * 16) + (dg & 3) * 4)
    slot = blk * 32 + sl * 4
    trs = tuple((row >> 6) * VTH + (row & 63) * 128 + (((slot >> 4) ^ (row & 7)) << 4) + (slot & 15) for row in (dg * 4 + r for r in range(4)))
    return (trl, trl1, trs), dg
T24 = sorted(task(w, l, r)[0] for w in range(8) for l in range(32) for r in range(2))
TWS = []
for h in range(2):
    for wi in range(4):
        for l in range(32):
            for r in range(2):
                pw = 2 * h + (wi & 1) + 4 * (wi >> 1)
                t, dg = task(pw, l, r)
                assert dg >> 4 == h, "warpgroup touches the other head"
                assert all(x // VTH == h for x in t[2]), "V^T store outside own head"
                # V source bytes: chunk dg >> 2 must be in head h's 64 bytes (chunks 4h..4h+3)
                assert (dg >> 2) // 4 == h
                TWS.append(t)
assert sorted(TWS) == T24 and len(set(TWS)) == 512
print("transpose tasks: same 512 tasks as v24, each warpgroup only its own head")
