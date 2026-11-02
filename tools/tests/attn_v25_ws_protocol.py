# Randomized interleaving model of attn_fvws's mbarrier / named-barrier protocol.
# Checks: (a) a stage is never overwritten while any consumer warp may still read it (K by S(i) wgmma until the warp's
# wait, V by the transpose); (b) consumers never read a stage before all 32 producer lanes' copies landed;
# (c) the V^T buffer (per head, i & 1) is never rewritten while P.V(i - 2) may still read it; (d) no deadlock.
import random
class MBar:
    def __init__(s, count): s.count, s.pending, s.phase = count, count, 0
    def arrive(s):
        s.pending -= 1
        assert s.pending >= 0
        if s.pending == 0: s.phase += 1; s.pending = s.count
    def done(s, parity): return (s.phase & 1) != parity  # try_wait.parity: phase with that parity completed
def run(ntiles, ST, seed):
    rnd = random.Random(seed)
    full = [MBar(32) for _ in range(ST)]; empty = [MBar(8) for _ in range(ST)]
    stage_tile = [None] * ST          # tile whose data lands / sits in the stage
    readers = [set() for _ in range(ST)]  # consumer warps that may still read the stage
    landed = [0] * ST
    inflight = []                     # (lane, stage) copies not yet landed (noinc arrive fires when landed)
    vt_readers = {}                   # (h, buf) -> set of warps whose P.V on it may be outstanding
    namedbar = {0: set(), 1: set()}
    procs = []
    def producer():
        stg = ph = 0
        for i in range(ntiles):
            if i >= ST:
                while not empty[stg].done(ph ^ 1): yield
            assert not readers[stg], f"stage {stg} overwritten while read by {readers[stg]}"
            stage_tile[stg] = i; landed[stg] = 0
            for lane in range(32): inflight.append((lane, stg)); yield
            stg += 1
            if stg == ST: stg = 0; ph ^= 1
    def lander():
        while True:
            if inflight and rnd.random() < 0.5:
                lane, stg = inflight.pop(rnd.randrange(len(inflight)))
                landed[stg] += 1; full[stg].arrive()
            yield
    def bar(h, w):
        gen = len(namedbar_gen[h])
        namedbar[h].add(w)
        if len(namedbar[h]) == 4: namedbar_gen[h].append(1); namedbar[h].clear()
        while len(namedbar_gen[h]) == gen: yield
    def consumer(w):
        h = w >> 2; st = ph = 0
        pv_out = None  # buffer my outstanding P.V reads
        for i in range(ntiles):
            while not full[st].done(ph): yield
            assert landed[st] == 32 and stage_tile[st] == i, "read before landing"
            yield from bar(h, w)
            readers[st].add(w)  # S(i) wgmma (K) + transpose (V) read the stage
            buf = i & 1
            assert not (vt_readers.get((h, buf), set()) - set()), f"V^T {h},{buf} rewritten while P.V reads it"
            yield
            yield from bar(h, w)
            # WG_WAIT_SO: S(i) and P.V(i-1) done for this warp... the wgmma is warpgroup-collective: model that the
            # P.V completes only once every warp has waited (worst case for the buffer hazard)
            if pv_out is not None: waited[(h, pv_out[0], pv_out[1])].add(w)
            readers[st].discard(w)
            empty[st].arrive()
            st += 1
            if st == ST: st = 0; ph ^= 1
            yield
            # P.V(i) issue reads V^T buf (until all 4 warps of the warpgroup waited on it)
            vt_readers.setdefault((h, buf), set()).add(w)
            waited.setdefault((h, i, buf), set())
            pv_out = (i, buf)
            yield
        if pv_out is not None: waited[(h, pv_out[0], pv_out[1])].add(w)
    namedbar_gen = {0: [], 1: []}
    waited = {}
    def janitor():  # a P.V on (h, buf) stops reading once all 4 warps waited
        while True:
            for (h, i, buf), ws in list(waited.items()):
                if len(ws) == 4:
                    vt_readers[(h, buf)] -= set(range(4 * h, 4 * h + 4)); del waited[(h, i, buf)]
            yield
    ps = [producer()] + [consumer(w) for w in range(8)]
    slow = rnd.choice(ps)
    alive = list(ps); bg = [lander(), janitor()]
    steps = 0
    while alive:
        steps += 1; assert steps < 10 ** 6, "deadlock"
        for g in bg: next(g)
        p = rnd.choice(alive)
        if p is slow and rnd.random() < 0.97: continue
        try: next(p)
        except StopIteration: alive.remove(p)
    return steps
n = 0
for ST in (2, 3, 4):
    for ntiles in (0, 1, 2, 3, 5, 9):
        for seed in range(150):
            run(ntiles, ST, seed); n += 1
print(f"protocol model: {n} randomized runs, no hazard / deadlock")
