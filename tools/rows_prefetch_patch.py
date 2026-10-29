"""Shared exact-text patch (decode v20 / attention v17): the split-KV "rows" kernel gets PREF (load the next tile's
page indices one iteration ahead, so each iteration waits on one DRAM round trip, the K / V gather, instead of two:
page index, then gather), and the batch >= 2 full-decode config moves to 128-key tiles with PREF on."""

LOOP_OLD = '''    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
        row = (page.to(tl.int64) * PAGE + t % PAGE) * W
'''
LOOP_NEW = '''    if PREF:  # the first tile's page indices; each iteration then loads the next tile's
        t0 = lo + tl.arange(0, BLOCK_N)
        pnext = tl.load(pt_ptr + b * stride_pt + t0 // PAGE, mask=t0 < hi, other=0)
    for n in range(lo, hi, BLOCK_N):
        t = n + tl.arange(0, BLOCK_N)
        tmask = t < hi
        if PREF:
            page = pnext
            tn = t + BLOCK_N
            pnext = tl.load(pt_ptr + b * stride_pt + tn // PAGE, mask=tn < hi, other=0)
        else:
            page = tl.load(pt_ptr + b * stride_pt + t // PAGE, mask=tmask, other=0)
        row = (page.to(tl.int64) * PAGE + t % PAGE) * W
'''
SIG_OLD = '''                     D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False):'''
SIG_NEW = '''                     D: tl.constexpr, HW: tl.constexpr, HOIST: tl.constexpr = False, PV16: tl.constexpr = False,
                     PREF: tl.constexpr = False):'''
CFG_OLD = '''    return ("rows", 64, 4, 3, 16, True)'''
CFG_NEW = '''    return ("rows", 128, 4, 3, 16, True)  # 128-key tiles + page-index prefetch (PREF = hoist), v20 / v17'''
CALL_OLD = '''BLOCK_M=rows, BLOCK_N=block_n, D=d,
            HW=True, HOIST=hoist,'''
CALL_NEW = '''BLOCK_M=rows, BLOCK_N=block_n, D=d,
            HW=True, HOIST=hoist, PREF=hoist,'''


def apply(s, rep, hw_text="True"):
    rep(LOOP_OLD, LOOP_NEW)
    rep(SIG_OLD, SIG_NEW)
    rep(CFG_OLD, CFG_NEW)
    rep(CALL_OLD.replace("HW=True", f"HW={hw_text}"), CALL_NEW.replace("HW=True", f"HW={hw_text}"))
