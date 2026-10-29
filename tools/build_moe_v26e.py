"""Builds muse_moe_w4a16/runs/v26e_nb2_evict.py from v26c_fused_chain.py (exact replacements, each asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_moe_w4a16/runs/v26c_fused_chain.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


def cut(a, b):
    """Remove s[index(a) : index(b)] (both asserted once)."""
    global s
    assert s.count(a) == 1 and s.count(b) == 1, (a[:50], b[:50])
    i, j = s.index(a), s.index(b)
    assert i < j
    s = s[:i] + s[j:]


rep('"""v26c = v26a with', '''"""v26e = v26c with:
- the CUDA tensor-core experts extended from <= 32 to <= 192 tokens, NB 2 work items (16 routes per expert chunk) at
  128 tokens (verify 16; it ran Marlin). In the decode layer the same kernels at 128 tokens took verify 16 from
  422 / 619 us (v17, Marlin) to 405 / 587 us (v18 check 9566de62f7e94ed1) and 402 / 594 us (v18b ae89a962123748e6).
  256 tokens stay on Marlin (NB 4 lost there: 115 registers, 2 CTAs/SM).
- weight and block-scale loads with an L2 evict-first cache policy and no L1 allocation (createpolicy +
  ld.global.nc.L1::no_allocate.L2::cache_hint), as Marlin does, so the FP32 sums, routing lists and activations
  stay L2-resident while ~0.1-0.7 GB of weights stream through once. Same values loaded: bit-identical to v26c.
- review round 10 minors: FASTSCALE per state (no module global), eager NVRTC compile of every launch config in
  prepare() (RawModule.compile()), and the exploration-only BENCH code removed.

v26c = v26a with''')
rep('''__device__ __forceinline__ u32 rotl(u32 w, int n) { return __funnelshift_l(w, w, n); }''',
'''__device__ __forceinline__ u32 rotl(u32 w, int n) { return __funnelshift_l(w, w, n); }
// streamed-once weight / scale loads: read-only path, no L1 allocation, L2 evict-first policy
// (one kernel-wide policy register: same registers / spills as v26c; making the policy inside each load spilled more)
__device__ __forceinline__ uint4 ldw(const uint4* p, unsigned long long pol) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ uint2 lds2(const uint2* p, unsigned long long pol) {
  uint2 v;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(pol));
  return v;
}''')
rep('''  for (int i = 0; i < PF; i++) if (i < NT) { qg[i] = __ldg(wg + i * 32); sgr[i] = __ldg(sg + i * 8); }''',
'''  unsigned long long pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  for (int i = 0; i < PF; i++) if (i < NT) { qg[i] = ldw(wg + i * 32, pol); sgr[i] = lds2(sg + i * 8, pol); }''')
rep('''  for (int i = 0; i < PF; i++) if (i < NT) { qu[i] = __ldg(wu + i * 32); sur[i] = __ldg(su + i * 8); }''',
'''  for (int i = 0; i < PF; i++) if (i < NT) { qu[i] = ldw(wu + i * 32, pol); sur[i] = lds2(su + i * 8, pol); }''')
rep('''    if (kb + PF < NT) { qg[kb % PF] = __ldg(wg + (kb + PF) * 32); sgr[kb % PF] = __ldg(sg + (kb + PF) * 8); }''',
'''    if (kb + PF < NT) { qg[kb % PF] = ldw(wg + (kb + PF) * 32, pol); sgr[kb % PF] = lds2(sg + (kb + PF) * 8, pol); }''')
rep('''    if (kb + PF < NT) { qu[kb % PF] = __ldg(wu + (kb + PF) * 32); sur[kb % PF] = __ldg(su + (kb + PF) * 8); }''',
'''    if (kb + PF < NT) { qu[kb % PF] = ldw(wu + (kb + PF) * 32, pol); sur[kb % PF] = lds2(su + (kb + PF) * 8, pol); }''')
rep('''#if GATED && WARPS == 8
#define MINB 4
#elif GATED && WARPS == 4
#define MINB 7
#else''', '''#if GATED && WARPS == 8 && NB == 1
#define MINB 4
#elif GATED && WARPS == 4 && NB == 1
#define MINB 7
#elif GATED && WARPS == 8 && NB == 2
#define MINB 3
#else''')
rep('''FASTSCALE = [0]  # set by prepare() from the weights' scale bytes


''', '')
rep('''def _mod(WARPS, KS, GATED, PF, NB):''', '''def _mod(WARPS, KS, GATED, PF, NB, FAST):''')
rep('''    key = (WARPS, KS, GATED, PF, NB, FASTSCALE[0])''', '''    key = (WARPS, KS, GATED, PF, NB, FAST)''')
rep('''f"-DFASTSCALE={FASTSCALE[0]}"))''', '''f"-DFASTSCALE={FAST}"))''')
rep('''    m1 = _mod(W1, H // S1, 1, P1, NB)
    m2 = _mod(W2, I // S2, 0, P2, NB)''', '''    m1 = _mod(W1, H // S1, 1, P1, NB, state["fast"])
    m2 = _mod(W2, I // S2, 0, P2, NB, state["fast"])''')
rep('''    FASTSCALE[0] = int(_scales_normal(weights["w13_sf"], start, count) and _scales_normal(weights["w2_sf"], start, count))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # compile the launch configs up front
        _mod(W1, H // S1, 1, P1, NB)
        _mod(W2, I // S2, 0, P2, NB)
    torch.cuda.synchronize()
    if BENCH:
        _bench(st, weights)
    return st''', '''    # FASTSCALE per state (r10): the PRMT + IMAD splat only when every local scale byte is a positive normal E4M3
    st["fast"] = fast = int(_scales_normal(weights["w13_sf"], start, count)
                            and _scales_normal(weights["w2_sf"], start, count))
    for NB, (W1, S1, P1), (W2, S2, P2) in SMALL_CFG.values():  # NVRTC-compile every launch config now (r10)
        _mod(W1, H // S1, 1, P1, NB, fast).compile()
        _mod(W2, I // S2, 0, P2, NB, fast).compile()
    torch.cuda.synchronize()
    return st''')
cut("# EXPLORATION ONLY (check runs): BENCH times launch configs", "# --- Marlin path (v9b_resubmit.py")
rep('''SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2))}''', '''# 128 (v26e): NB 2 items, 16 routes per expert chunk (decode v18 / v18b measured the gain at 128 tokens).
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2))}''')
rep('''SMALL_MAX = 32  # token counts up to this use the tensor-core CUDA path; larger ones use Marlin (v9b)''',
    '''SMALL_MAX = 192  # token counts up to this use the tensor-core CUDA path; larger ones use Marlin (v9b)''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_moe_w4a16/runs/v26e_nb2_evict.py"
dst.write_text(s)
