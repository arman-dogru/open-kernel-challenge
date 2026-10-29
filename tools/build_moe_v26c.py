"""Builds muse_moe_w4a16/runs/v26c_fused_chain.py from v26a_int_splat.py (exact replacements, each asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_moe_w4a16/runs/v26a_int_splat.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('"""v26a = v24 with', '''"""v26c = v26a with the small-batch launch chain cut from 6 kernels to 4 (bit-identical outputs):
- the FP32 sum buffers are torch.empty (no zero-fill kernel): route_prep runs one block per route and zeroes that
  route's GEMM-1 (2I) and GEMM-2 (H) FP32 rows if the route is local (block 0 also builds the lists, as before);
- silu_mul is gone: GEMM 1's split-K CTAs count arrivals per (work item, row block) in a self-resetting int32
  counter (state["c1"], zero at prepare, reset by its last CTA), and the last CTA of each (item, row block) applies
  silu(gate) * up to its 16 * WARPS rows of the item's routes and writes them as BF16, with silu_mul's expression.
Launches: route_prep, moe_mma (GEMM 1 + SiLU), moe_mma (GEMM 2), post-norm sum.

v26a = v24 with''')
rep('''extern "C" __global__ void route_prep(const int* __restrict__ ids, int* slot, int* cnt, int* lst, int* work,
                                      int* nwork, int NR, int local_start, int maxm, int mb) {
  __shared__ int scnt[32];''', '''// grid (NR): every block zeroes its route's FP32 GEMM-1 (RW1 floats) and GEMM-2 (RW2 floats) sum rows if the route
// is local; block 0 also builds the route lists.
extern "C" __global__ void route_prep(const int* __restrict__ ids, int* slot, int* cnt, int* lst, int* work,
                                      int* nwork, int NR, int local_start, int maxm, int mb,
                                      float* __restrict__ h, int RW1, float* __restrict__ y, int RW2) {
  {
    const int e = ids[blockIdx.x] - local_start;
    if (e >= 0 && e < 32) {
      float4* hr = reinterpret_cast<float4*>(h + (size_t)blockIdx.x * RW1);
      for (int c = threadIdx.x; c < RW1 / 4; c += blockDim.x) hr[c] = make_float4(0.f, 0.f, 0.f, 0.f);
      float4* yr = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * RW2);
      for (int c = threadIdx.x; c < RW2 / 4; c += blockDim.x) yr[c] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  if (blockIdx.x != 0) return;
  __shared__ int scnt[32];''')
rep('''    int maxm, int local_start, int N, int K, int xrow_div) {
  __shared__ __align__(16) u16 xs[8 * NB * XS];''', '''    int maxm, int local_start, int N, int K, int xrow_div, u16* __restrict__ inter, int* __restrict__ c1) {
  __shared__ __align__(16) u16 xs[8 * NB * XS];
  __shared__ int last;''')
rep('''        atomicAdd(out + (size_t)r * OUTW + N + rr, cu[nb][c] * gsc);
#endif
      }
    }
}''', '''        atomicAdd(out + (size_t)r * OUTW + N + rr, cu[nb][c] * gsc);
#endif
      }
    }
#if GATED
  // split-K arrival count for (work item, row block); the last CTA writes silu(gate) * up as BF16 (silu_mul's math)
  __threadfence();
  __syncthreads();
  int* ctr = c1 + (size_t)wk * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) last = (atomicAdd(ctr, 1) == (int)gridDim.z - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int nr = min(8 * NB, ne - i0), r0 = blockIdx.x * WARPS * 16;
  for (int idx = threadIdx.x; idx < nr * WARPS * 16; idx += blockDim.x) {
    const int n = idx / (WARPS * 16), rr = r0 + idx % (WARPS * 16);
    const int r = lst[ex * maxm + i0 + n];
    const float gv = __ldcg(out + (size_t)r * OUTW + rr), uv = __ldcg(out + (size_t)r * OUTW + N + rr);
    inter[(size_t)r * N + rr] = f2bf(gv / (1.f + __expf(-gv)) * uv);
  }
  if (threadIdx.x == 0) *ctr = 0;
#endif
}''')
rep('''extern "C" __global__ void __launch_bounds__(32 * WARPS) moe_mma(''', '''// GATED: keep v26a's CTAs / SM (8 warps: 4, 64 registers; 4 warps: 7, <= 73); the SiLU epilogue alone would push
// them to 70 / 83 registers.
#if GATED && WARPS == 8
#define MINB 4
#elif GATED && WARPS == 4
#define MINB 7
#else
#define MINB 1
#endif
extern "C" __global__ void __launch_bounds__(32 * WARPS, MINB) moe_mma(''')
rep('''    buf = torch.zeros((NR * (2 * I + H),), device=dev, dtype=torch.float32)''',
    '''    buf = torch.empty((NR * (2 * I + H),), device=dev, dtype=torch.float32)  # local rows zeroed by route_prep''')
rep('''        m1.get_function("route_prep")((1,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                     np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(MB)))
        m1.get_function("moe_mma")((I // 16 // W1, nwork, S1), (32 * W1,), (
            _p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(h32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H), np.int32(top_k)))
        m1.get_function("silu_mul")((NR,), (256,), (_p(h32), _p(inter), _p(slot), np.int32(I)))
        m2.get_function("moe_mma")((H // 16 // W2, nwork, S2), (32 * W2,), (
            _p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(y32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I), np.int32(1)))''',
'''    c1 = state["c1"]
    assert nwork * (I // 16 // W1) <= c1.numel()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((NR,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                      np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(MB),
                                                      _p(h32), np.int32(2 * I), _p(y32), np.int32(H)))
        m1.get_function("moe_mma")((I // 16 // W1, nwork, S1), (32 * W1,), (
            _p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(h32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H), np.int32(top_k),
            _p(inter), _p(c1)))
        m2.get_function("moe_mma")((H // 16 // W2, nwork, S2), (32 * W2,), (
            _p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(y32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I), np.int32(1),
            np.uint64(0), np.uint64(0)))''')
rep('''    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
    c1 = state["c1"]''', '''    c1 = state["c1"]''')
rep('''    st.update({
        "w13": w13,''', '''    st.update({
        "c1": torch.zeros((8192,), device=weights["w13"].device, dtype=torch.int32),  # GEMM-1 split-K counters
        "w13": w13,''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_moe_w4a16/runs/v26c_fused_chain.py"
dst.write_text(s)
