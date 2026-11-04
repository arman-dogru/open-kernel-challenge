import itertools
T=[6,7,8,15,22,23,24,31]
def sh(S,r): return frozenset((p-r)%32 for p in S)
ALL=frozenset(range(32))
def two_piece(rest,u,v):
    # can rest = {t-u for t in M1} U {t-v for t in T\M1}, bijectively?
    for k in range(9):
        for M1 in itertools.combinations(T,k):
            M2=[t for t in T if t not in M1]
            s=sh(M1,u)|sh(M2,v)
            if len(s)==8 and s==rest: return M1
    return None
hits=[]
# case A: 3 single regs (rotations x<y<z, any incl 0) + one 2-piece reg (u,v); nonzero(R)<=3
for x,y,z in itertools.combinations(range(32),3):
    S=[sh(T,x),sh(T,y),sh(T,z)]
    if len(S[0]|S[1]|S[2])!=24: continue
    rest=ALL-(S[0]|S[1]|S[2])
    for u,v in itertools.combinations(range(32),2):
        R={x,y,z,u,v}-{0}
        if len(R)+5>8: continue
        m=two_piece(rest,u,v)
        if m is not None: hits.append(("A",x,y,z,u,v,m))
# case B: 2 single + 2 two-piece regs, nonzero(R)<=2
for x,y in itertools.combinations(range(32),2):
    S=sh(T,x)|sh(T,y)
    if len(S)!=16: continue
    for a,b in itertools.combinations(range(32),2):
        R={x,y,a,b}-{0}
        if len(R)>2: continue
        rest=ALL-S
        # split rest into two regs each two-piece over rotations a,b (or x,y)
        rots=sorted({0,x,y,a,b})
        for u1,v1,u2,v2 in itertools.product(rots,repeat=4):
            if len({u1,v1,u2,v2}-{0}|R)>2: continue
            for k in range(9):
                for M1 in itertools.combinations(T,k):
                    M2=[t for t in T if t not in M1]
                    s1=sh(M1,u1)|sh(M2,v1)
                    if len(s1)!=8 or not s1<=rest: continue
                    if two_piece(rest-s1,u2,v2) is not None: hits.append(("B",x,y,u1,v1,u2,v2,M1))
print(len(hits)); print(hits[:5])
