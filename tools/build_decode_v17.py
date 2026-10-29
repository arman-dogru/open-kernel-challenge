"""Builds muse_decode_layer/runs/v17_fused_chain.py from v16_int_splat.py (exact replacements, each asserted once):
MoE v26c's launch-chain fusion in the layer's CUDA expert path (the CUDA-side replacements are muse_moe_w4a16
v26c's, tools/build_moe_v26c.py on branch moe-v26c, verbatim: the moe_mma / route_prep sources are identical in
both files), plus the CUDA experts at 17-32 tokens. Run at HEAD; the committed file must be byte-identical to its
output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_decode_layer/runs/v16_int_splat.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''# okc: problem=muse_decode_layer
"""v16:''', '''# okc: problem=muse_decode_layer
"""v17: v16 plus muse_moe_w4a16 v26c's launch-chain fusion in the CUDA expert path (check f479a34c48984e24, 1.497x
there vs v26a 1.457x), 6 -> 4 launches with bit-identical math: the FP32 sum buffers are torch.empty and route_prep
(one block per route) zeroes the local routes' GEMM-1 / GEMM-2 rows; silu_mul is folded into GEMM 1: its split-K
CTAs count arrivals per (work item, row block) in self-resetting int32 counters (state["moe_small"]["c1"], zeroed
in prepare, reset to 0 by the last CTA inside the same launch, so every call and CUDA-graph replay starts from
zeros), and the last CTA writes silu(gate) * up as BF16 with silu_mul's expression. The CUDA experts now also run
17-32 tokens (decode 32) with the standalone MoE's 32-token config (NB 1, 8 warps, split 4, prefetch 2): v15b had
them ~6-11 us behind Marlin there before v26a's splat (~-10 us at bs32) and v26c's chain (~-10 us).

v16:''')
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

rep("""    buf = torch.zeros((NR * (2 * I + H),), device=dev, dtype=torch.float32)""",
    """    buf = torch.empty((NR * (2 * I + H),), device=dev, dtype=torch.float32)  # local rows zeroed by route_prep""")
rep("""    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        m1.get_function("route_prep")((1,), (256,), (_p(ids), _p(slot), _p(cnt), _p(lst), _p(work), _p(nw),
                                                     np.int32(NR), np.int32(start), np.int32(MAXM), np.int32(MB)))
        m1.get_function("moe_mma")((I // 16 // W1, nwork, S1), (32 * W1,), (
            _p(x), _p(state["w13"]), _p(state["w13_sf"]), _p(state["w13_gs"]), _p(h32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(I), np.int32(H), np.int32(top_k)))
        m1.get_function("silu_mul")((NR,), (256,), (_p(h32), _p(inter), _p(slot), np.int32(I)))
        m2.get_function("moe_mma")((H // 16 // W2, nwork, S2), (32 * W2,), (
            _p(inter), _p(state["w2"]), _p(state["w2_sf"]), _p(state["w2_gs"]), _p(y32), _p(cnt), _p(lst),
            _p(work), _p(nw), np.int32(MAXM), np.int32(start), np.int32(H), np.int32(I), np.int32(1)))""",
    """    c1 = state["c1"]
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
            np.uint64(0), np.uint64(0)))""")
rep("""    st = {"w13": w13, "w13_sf": w13_sf,""",
    """    st = {"c1": torch.zeros((8192,), device=w13.device, dtype=torch.int32),  # GEMM-1 split-K counters (self-resetting)
          "w13": w13, "w13_sf": w13_sf,""")
rep("""        for name in ("route_prep", "moe_mma", "silu_mul"):""", """        for name in ("route_prep", "moe_mma"):""")
rep("""# The layer uses these CUDA experts up to LAYER_SMALL_MAX = 16 tokens only, so v24's 32-token entry is not carried.
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2))}""",
    """# 32: the standalone MoE's 32-token entry (v23 bench, judge routing; v15a's layer bench: 8 and 16 warps tied).
SMALL_CFG = {1: (1, (4, 8, 2), (4, 8, 2)), 8: (1, (8, 4, 2), (8, 4, 2)), 16: (1, (8, 4, 2), (8, 4, 2)),
             32: (1, (8, 4, 2), (8, 4, 2))}""")
rep("LAYER_SMALL_MAX = 16  # tokens up to this use the CUDA experts (v9 judge numbers)",
    "LAYER_SMALL_MAX = 32  # tokens up to this use the CUDA experts (v9 judge numbers to 16; v17: 17-32, see docstring)")
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_decode_layer/runs/v17_fused_chain.py"
dst.write_text(s)
