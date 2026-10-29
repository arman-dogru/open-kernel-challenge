"""Build muse_prefill_layer/runs/v5_dequant_persist.py from v3_gs_after_local.py (byte-exact, rerunnable).

v5 = v3 with the per-call NVFP4 -> BF16 expert dequant (v3 check efae26e3adb24c0f: 1.55 ms per first-turn shape for
~0.8 GB read + 3 GB written, ~2.5 TB/s) as fewer, larger programs: a persistent grid (NPROG = 132 * 8 programs, each
looping over rows pid, pid + NPROG, ...) instead of one program per row (393,216 programs), and each packed byte's
two BF16 values stored as one uint32 (low element in the low half) so loads and stores stay contiguous with no
tl.join / reshape interleave (an offline sm_90 compile shows v3's join costs 1 KB shared + a barrier per row) (v4's 4-row tiles of that join measured slower, 1.89 ms, check 04328d59e9dc4916).
Same exact values: e2m1 x e4m3 in FP32, rounded once to BF16 (exact), tensor scale after the GEMMs as v3.
"""
import sys
SRC, DST = "muse_prefill_layer/runs/v3_gs_after_local.py", "muse_prefill_layer/runs/v5_dequant_persist.py"
s = open(SRC).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


a = s.index("@triton.jit\ndef _nvfp4_dequant_kernel(")
b = s.index("@triton.jit\ndef _gather_local_kernel(")
sub(s[a:b], '''@triton.jit
def _nvfp4_dequant_kernel(q_ptr, sf_ptr, out_ptr, n_rows, ROW_BYTES: tl.constexpr, NPROG: tl.constexpr):
    # uint8 [E, N, K/2] (element 2i in the low nibble) with e4m3 [E, N, K/16] -> BF16 [E, N, K] of e2m1 x e4m3, which
    # BF16 holds exactly (at most 5 significant bits); the FP32 tensor scale is applied after the GEMMs (v3, review
    # of #16 m1). v5: persistent programs over rows pid, pid + NPROG, ...; byte i's two BF16 values leave as one
    # uint32 (element 2i in the low half, little-endian) of out viewed as int32 [E, N, K/2], so no join / layout
    # conversion and the loads / stores stay contiguous.
    c = tl.arange(0, ROW_BYTES)
    pid = tl.program_id(0)
    first = pid.to(tl.int64)
    for i in range(0, (n_rows - pid + NPROG - 1) // NPROG):
        row = first + i * NPROG
        b = tl.load(q_ptr + row * ROW_BYTES + c).to(tl.int32)
        sfb = tl.load(sf_ptr + row * (ROW_BYTES // 8) + c // 8)  # e4m3 scale bytes as uint8
        scale = sfb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        lo = b & 0xF
        hi = (b >> 4) & 0xF
        mlo = lo & 7
        mhi = hi & 7
        # E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 for codes 0..7
        vlo = tl.where(mlo < 4, mlo.to(tl.float32) * 0.5, tl.where(mlo < 6, (mlo - 2).to(tl.float32),
                       tl.where(mlo == 6, 4.0, 6.0)))
        vhi = tl.where(mhi < 4, mhi.to(tl.float32) * 0.5, tl.where(mhi < 6, (mhi - 2).to(tl.float32),
                       tl.where(mhi == 6, 4.0, 6.0)))
        vlo = tl.where((lo & 8) != 0, -vlo, vlo) * scale
        vhi = tl.where((hi & 8) != 0, -vhi, vhi) * scale
        blo = vlo.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        bhi = vhi.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        tl.store(out_ptr + row * ROW_BYTES + c, (blo | (bhi << 16)).to(tl.int32, bitcast=True))


def _dequant(q, sf):
    e, n, kb = q.shape
    out = torch.empty((e, n, 2 * kb), device=q.device, dtype=torch.bfloat16)
    nprog = min(e * n, 132 * 8)
    _nvfp4_dequant_kernel[(nprog,)](q, sf.view(torch.uint8), out.view(torch.int32), e * n, ROW_BYTES=kb, NPROG=nprog,
                                    num_warps=8)
    return out


''')
open(sys.argv[1] if len(sys.argv) > 1 else DST, "w").write(s)
print("ok")
