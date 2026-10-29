"""Builds muse_decode_layer/runs/v19_nb2_256_evict.py from v18b_cuda_experts_192.py (exact replacements, each asserted
once). Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v18b_cuda_experts_192.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''# okc: problem=muse_decode_layer
"""v18b:''', '''# okc: problem=muse_decode_layer
"""v19: v18b with
- the CUDA experts also at 256 tokens (verify 32), with NB 2 items (16 routes per expert chunk) like at 128. v18 tried
  NB 4 there and lost (115 registers, 2 CTAs/SM); NB 2 has 80 registers / 3 CTAs per SM. In v18b's profile Marlin
  took 424 / 450 us at verify 32. The item model (~25 experts, ~40 NB 2 items, vs ~30 items at 128 tokens, where
  both expert GEMMs took ~275 us) projects ~350-370 us.
- the weight / block-scale loads of the expert kernels with an L2 evict-first policy and no L1 allocation, as MoE v26e
  (board 1.539x) has them. Same values loaded.

v18b:''')
rep('''__device__ __forceinline__ u32 rotl(u32 w, int n) { return __funnelshift_l(w, w, n); }''',
'''__device__ __forceinline__ u32 rotl(u32 w, int n) { return __funnelshift_l(w, w, n); }
// streamed-once weight / scale loads: read-only path, no L1 allocation, L2 evict-first policy
// (one kernel-wide policy register: same registers / spills as without; as MoE v26e)
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
rep('''             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2))}''',
    '''             32: (1, (8, 4, 2), (8, 4, 2)), 128: (2, (8, 4, 2), (8, 4, 2)), 256: (2, (8, 4, 2), (8, 4, 2))}''')
rep("LAYER_SMALL_MAX = 192  # tokens up to this use the CUDA experts (v17: <= 32; v18b: <= 192, NB 2 above 64)",
    "LAYER_SMALL_MAX = 256  # tokens up to this use the CUDA experts (v17: <= 32; v19: <= 256, NB 2 above 64)")
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v19_nb2_256_evict.py"
dst.write_text(s)
