T=[6,7,8,15,22,23,24,31]
# source sets in Z64 for one register: in-place A or B, rotl within A or B, or funnel of A:B / B:A by n -> T+off cyclic in Z64
tiles=set()
for off in range(64): tiles.add(frozenset((t+off)%64 for t in T))
for w in (0,32):
    for r in range(32): tiles.add(frozenset(w+((t-r)%32) for t in T))
tiles=list(tiles)
by=[[t for t in tiles if c in t] for c in range(64)]
sol=[]
def rec(cov,ch):
    if len(cov)==64: sol.append(list(ch)); return True
    c=min(i for i in range(64) if i not in cov)
    for t in by[c]:
        if t & cov: continue
        ch.append(t)
        if rec(cov|t,ch): return True
        ch.pop()
    return False
print(rec(frozenset(),[]), sol[:1])
