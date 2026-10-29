"""Builds muse_moe_w4a16/runs/v26d_async_stage.py from v26c_fused_chain.py (exact replacements, each asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_moe_w4a16/runs/v26c_fused_chain.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""v26c = v26a with', '''"""v26d (offline only) = v26c with moe_mma's activation staging issued as cp.async.cg 16-byte copies straight into
shared memory (zero-filled by src-size 0 past the chunk's routes) instead of LDG -> register -> STS, one commit
group, waited before the existing barrier. Same bytes in the same smem slots: bit-identical to v26c.

v26c = v26a with''')
rep('''  for (int idx = threadIdx.x; idx < 8 * NB * (KS / 8); idx += blockDim.x) {
    int m = idx / (KS / 8), c = idx % (KS / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (i0 + m < ne) {
      int r = lst[ex * maxm + i0 + m];
      v = reinterpret_cast<const uint4*>(xin + (size_t)(r / xrow_div) * K + k0)[c];
    }
    reinterpret_cast<uint4*>(xs + m * XS)[c] = v;
  }
  __syncthreads();''', '''  // cp.async.cg (L2 only, 16 B): src-size 0 zero-fills the rows past the chunk's routes
  for (int idx = threadIdx.x; idx < 8 * NB * (KS / 8); idx += blockDim.x) {
    int m = idx / (KS / 8), c = idx % (KS / 8);
    const uint4* src = reinterpret_cast<const uint4*>(xin);  // any valid address when nothing is read
    unsigned sz = 0;
    if (i0 + m < ne) {
      int r = lst[ex * maxm + i0 + m];
      src = reinterpret_cast<const uint4*>(xin + (size_t)(r / xrow_div) * K + k0) + c;
      sz = 16;
    }
    unsigned dst = (unsigned)__cvta_generic_to_shared(xs + m * XS) + c * 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(sz) : "memory");
  }
  asm volatile("cp.async.commit_group;\\n\\tcp.async.wait_group 0;" ::: "memory");
  __syncthreads();''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_moe_w4a16/runs/v26d_async_stage.py"
dst.write_text(s)
