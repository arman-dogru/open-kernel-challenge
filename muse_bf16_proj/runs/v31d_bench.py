# okc: problem=muse_bf16_proj
"""v31d (DIAGNOSTIC, not for the board): v6's kernel_fn unchanged, plus the v31 TMA-bulk weight-streaming CUDA
kernel (tools/bf16_v31_stream.cu) compiled in prepare() and an untimed one-time sweep of its candidates against v6
and cuBLAS for every BF16 layer x m <= 32 and the router (tools/build_bf16_v31.py --diag).

v6: Dense projections: a swap-AB Triton GEMM for small m (weights as the MMA's M side, tokens
padded to 16 as its N side), with split-K + FP32 atomics for the narrow router.

- BF16 layers: out^T[n, m] = W[n, :] . x[m, :], FP32 accumulation, BF16 store. Weights untouched.
- Router: FP32 weights as a BF16 hi + lo pair (allowed by the rules at any m), both dotted
  against the same x tile in one pass, FP32 accumulation, split-K with FP32 atomic adds.
- Router above 32 tokens: the same hi + lo products as SGLang's two cuBLAS GEMMs, run as one
  cuBLAS GEMM against the stacked [hi; lo] weights (FP32 out), then hi-part + lo-part in FP32.
- Where cuBLAS won in H200 check runs (qkv_gate, unpooler, and every other layer above 32
  tokens) SGLang's own path is kept."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _swapab_gemm(x_ptr, w_ptr, lo_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr,
                 HAS_LO: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    k_per = K // SPLIT_K
    k0 = pid_k * k_per
    n_mask = offs_n < N
    m_mask = offs_m < M
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + (k0 + offs_k)[:, None]
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        xt = tl.load(x_ptrs + kk, mask=m_mask[None, :], other=0.0)
        w = tl.load(w_ptrs + kk, mask=n_mask[:, None], other=0.0)
        acc = tl.dot(w, xt, acc)
        if HAS_LO:
            lo = tl.load(lo_ptr + offs_n[:, None] * stride_wn + (k0 + kk + offs_k)[None, :],
                         mask=n_mask[:, None], other=0.0)
            acc = tl.dot(lo, xt, acc)
    o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    o_mask = m_mask[None, :] & n_mask[:, None]
    if SPLIT_K > 1:
        tl.atomic_add(o_ptrs, acc, mask=o_mask, sem="relaxed")
    elif OUT_BF16:
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)
    else:
        tl.store(o_ptrs, acc, mask=o_mask)


# (BN, BK, SPLIT_K, num_warps, num_stages) per layer, for m <= 32, picked per shape from H200
# check runs (v1/v2 in runs/). Layers set to None keep SGLang's cuBLAS path, which won there.
CONFIGS = {
    "qkv_gate": None,
    "o": (64, 128, 1, 4, 4),
    "pooler": (32, 256, 1, 4, 4),
    "unpooler_s": None,
    "unpooler": None,
    "router": (16, 256, 8, 4, 4),
}
# Above this m, cuBLAS (SGLang's path) wins on every layer so far.
TRITON_MAX_M = 32


def _bm(m):
    return 16 if m <= 16 else (32 if m <= 32 else 64)



# ---------------- v31: TMA-bulk weight-streaming CUDA kernel (tools/bf16_v31_stream.cu, tools/bf16_v31_cfg.py) -------
_V31_SRC = r"""
// bf16 v31: weight-streaming skinny GEMM out[m, n] = x[m, K] . W[n, K]^T for m <= 32 (BF16 in, FP32 accumulate,
// BF16 out; FP32 out for the router's hi/lo pair), sm_90a, mma.sync m16n8k16 swap-AB (W rows = MMA M, tokens = N).
// Why not v30: v30's register loads (LDG.128 interleaved with their MMAs by ptxas, ~2 per warp in flight) and an
// unbalanced grid (qkv_gate: 144 rowtiles on 132 SMs) kept it at ~2.6-2.9 TB/s. v31 instead:
// - one producer warp per CTA streams W with TMA bulk copies (cp.async.bulk, L2 evict-first) into an ST-deep ring of
//   SPS-KB stages (64-128 KB in flight per SM), mbarrier full/empty handshakes, NW consumer warps read A fragments
//   from shared memory (contiguous 16 B per lane, conflict-free) and run the MMAs.
// - W repacked (prepare) into A-fragment order [rowtile (16 rows)][pair (32 k)][step s][lane][8 bf16] (the v30
//   layout); k permuted inside each pair (lane t, step s holds k = 8t + 4s + 0..3) so x's B fragments for both steps
//   are one 16 B shared-memory load.
// - balanced 1-CTA-per-SM grid of exactly (n/16/G) * CS CTAs (<= 132): a cluster of CS CTAs owns G rowtiles; rank r
//   owns k-slice r (K/CS) of all G rowtiles, so its x slice is bulk-copied into shared memory once and reused G times.
// - split-K reduction without atomics, fixed order: consumer warps -> shared memory per rowtile, then the CS ranks'
//   partials through distributed shared memory (each output rowtile summed by one owner rank, ranks in order).
//   HL = 2 sums rowtile pairs (router: rowtile 2c = hi rows, 2c+1 = lo rows of output rowtile c). Deterministic.
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;

__device__ __forceinline__ void mma(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even (finite)
  u32 u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (u16)(u >> 16);
}
__device__ __forceinline__ u32 sa(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u32 cluster_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_dsmem(const float* p, u32 rank) {
  u32 ra;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(sa(p)), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}
__device__ __forceinline__ void mbar_init(u64* b, u32 count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(u64* b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u64* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(u64* b, u32 parity) {  // traps after ~2^26 failed polls (never hang)
  u32 ok = 0, n = 0;
  while (true) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(sa(b)), "r"(parity) : "memory");
    if (ok) break;
    if (++n > (1u << 26)) asm volatile("trap;");
  }
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, u32 bytes, u64* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s_ef(void* dst, const void* src, u32 bytes, u64* bar, u64 pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(sa(p))
               : "memory");
  return v;
}
__device__ __forceinline__ void bar_consumers(int nthreads) {
  asm volatile("bar.sync 1, %0;" :: "r"(nthreads) : "memory");
}

// Shared memory (dynamic, host computes the same sizes in bf16_v31_cfg.py smem()):
//   ring [ST][SPS KB] | xs [8 NT rows][XROW = Pc*64 + 64 B] | red [2][NW][E] f32 | seg [G][E] f32 (CS > 1 or HL > 1)
//   | mbarriers full[ST], empty[ST], x
// Launch: grid (n_rows/16/G) * CS CTAs, (NW + 1) * 32 threads. Pc = K/32/CS pairs per rank per rowtile.
template <int NT, int CS, int NW, int ST, int SPS, int HL, bool F32>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__((NW + 1) * 32, 1)
bf16_v31(const unsigned char* __restrict__ W, const u16* __restrict__ x, int ldx_, void* __restrict__ out, int m,
         int n_out, int K, int G) {
  constexpr int E = NT * 128;  // FP32 partials per rowtile: 16 rows x 8 NT tokens
  constexpr bool SEG = CS > 1 || HL > 1;
  constexpr u32 SB = SPS * 1024;
  extern __shared__ __align__(128) unsigned char sm[];
  const int PPR = K >> 5, Pc = PPR / CS, nst = Pc / SPS;
  const int XROW = Pc * 64 + 64;
  unsigned char* ring = sm;
  unsigned char* xs = sm + ST * SB;
  float* red = reinterpret_cast<float*>(xs + 8 * NT * XROW);
  float* seg = red + 2 * NW * E;
  u64* bars = reinterpret_cast<u64*>(seg + (SEG ? G * E : 0));
  u64* full = bars;
  u64* empty = bars + ST;
  u64* xbar = bars + 2 * ST;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const u32 rank = CS > 1 ? cluster_rank() : 0u;
  const int cl = blockIdx.x / CS;
  const int rt0 = cl * G;              // first rowtile of this cluster
  const int kp0 = (int)rank * Pc;      // first pair of this rank's k-slice
  if (threadIdx.x == 0) {
    for (int i = 0; i < ST; ++i) {
      mbar_init(full + i, 1);
      mbar_init(empty + i, NW);
    }
    mbar_init(xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == NW) {  // producer
    if (lane == 0) {
      u64 pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      mbar_expect_tx(xbar, (u32)m * (u32)(Pc * 64));
      for (int r = 0; r < m; ++r)
        bulk_g2s(xs + r * XROW, x + (size_t)r * ldx_ + kp0 * 32, (u32)(Pc * 64), xbar);
      int it = 0;
      for (int g = 0; g < G; ++g) {
        const unsigned char* src = W + ((size_t)(rt0 + g) * PPR + kp0) * 1024;
        for (int c = 0; c < nst; ++c, ++it) {
          const int s = it % ST;
          if (it >= ST) mbar_wait(empty + s, (u32)((it / ST) - 1) & 1u);
          mbar_expect_tx(full + s, SB);
          bulk_g2s_ef(ring + s * SB, src + (size_t)c * SB, SB, full + s, pol);
        }
      }
    }
  } else {  // consumers
    const int t = lane & 3, gq = lane >> 2;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
    mbar_wait(xbar, 0);
    const unsigned char* xl = xs + gq * XROW + 16 * t;
    int it = 0;
    for (int g = 0; g < G; ++g) {
      for (int c = 0; c < nst; ++c, ++it) {
        const int s = it % ST;
        mbar_wait(full + s, (u32)(it / ST) & 1u);
        const unsigned char* A = ring + s * SB + lane * 16;
#pragma unroll 4
        for (int i = warp; i < SPS; i += NW) {
          const uint4 a0 = lds128(A + i * 1024), a1 = lds128(A + i * 1024 + 512);
          const unsigned char* xb = xl + (c * SPS + i) * 64;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint4 b = lds128(xb + j * 8 * XROW);
            mma(acc[j], a0.x, a0.y, a0.z, a0.w, b.x, b.y);
            mma(acc[j], a1.x, a1.y, a1.z, a1.w, b.z, b.w);
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + s);
      }
      // rowtile g done: reduce the NW warps' partials in fixed order (double-buffered: one barrier per rowtile)
      float* rb = red + (g & 1) * NW * E;
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          rb[warp * E + (j * 4 + c) * 32 + lane] = acc[j][c];
          acc[j][c] = 0.f;
        }
      bar_consumers(NW * 32);
      for (int e = threadIdx.x; e < E; e += NW * 32) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) v += rb[w * E + e];
        if (SEG) {
          seg[g * E + e] = v;
        } else {
          const int ln = e & 31, c = (e >> 5) & 3, j = e >> 7;
          const int row = (rt0 + g) * 16 + (ln >> 2) + 8 * (c >> 1);
          const int col = 8 * j + 2 * (ln & 3) + (c & 1);
          if (col < m) {
            if (F32) reinterpret_cast<float*>(out)[(size_t)col * n_out + row] = v;
            else reinterpret_cast<u16*>(out)[(size_t)col * n_out + row] = f2bf(v);
          }
        }
      }
    }
  }
  __syncwarp();
  if (SEG) {
    if (CS > 1) cluster_sync(); else __syncthreads();
    const int GO = G / HL;  // output rowtiles of this cluster
    for (int o = (int)rank; o < GO; o += CS) {
      for (int e = threadIdx.x; e < E; e += (NW + 1) * 32) {
        float v = 0.f;
#pragma unroll
        for (int r = 0; r < CS; ++r)
#pragma unroll
          for (int h = 0; h < HL; ++h) v += CS > 1 ? ld_dsmem(seg + (o * HL + h) * E + e, (u32)r) : seg[(o * HL + h) * E + e];
        const int ln = e & 31, c = (e >> 5) & 3, j = e >> 7;
        const int row = (cl * GO + o) * 16 + (ln >> 2) + 8 * (c >> 1);
        const int col = 8 * j + 2 * (ln & 3) + (c & 1);
        if (col < m) {
          if (F32) reinterpret_cast<float*>(out)[(size_t)col * n_out + row] = v;
          else reinterpret_cast<u16*>(out)[(size_t)col * n_out + row] = f2bf(v);
        }
      }
    }
    if (CS > 1) cluster_sync();
  }
}
"""
# tools/bf16_v31_cfg.py (body; names prefixed _v31 / V31_)

V31_SMS = 132
V31_SMEM_MAX = 227 * 1024
V31_SMEM_MIN = 116 * 1024  # request at least this much so two CTAs never share an SM
# key -> (rows in the repacked weight, K, HL, F32, n_out); router rows are [hi 16 rows, lo 16 rows] per output rowtile
V31_LAYERS = {"qkv_gate": (2304, 8192, 1, False, 2304), "o": (8192, 1024, 1, False, 8192),
          "pooler": (4096, 8192, 1, False, 4096), "unpooler": (8192, 4096, 1, False, 8192),
          "unpooler_s": (1024, 4096, 1, False, 1024), "router": (512, 8192, 2, True, 256)}
V31_PLANS = {"qkv_gate": [(8, 9)], "o": [(1, 4), (2, 8), (4, 16)], "pooler": [(8, 16), (4, 8), (2, 4)],
         "unpooler": [(4, 16), (8, 32), (2, 8)], "unpooler_s": [(2, 1), (4, 2), (8, 4)], "router": [(8, 2)]}
V31_TUNES = [(4, 4, 32), (8, 4, 32), (4, 8, 16), (4, 3, 32), (4, 6, 16), (8, 3, 32)]  # (NW consumer warps, ST stages, SPS pairs (KB) per stage)
V31_NTS = (1, 2, 4)


def _v31_smem(inst, K, G):
    nt, cs, nw, st, sps, hl, f32 = inst
    E = nt * 128
    pc = K // 32 // cs
    seg = cs > 1 or hl > 1
    return st * sps * 1024 + 8 * nt * (pc * 64 + 64) + 2 * nw * E * 4 + (G * E * 4 if seg else 0) + (2 * st + 1) * 8


def _v31_valid(key, nt, plan, tune):
    rows, K, hl, f32, _ = V31_LAYERS[key]
    cs, G = plan
    nw, st, sps = tune
    pc = K // 32 // cs
    nrt = rows // 16
    return (K // 32) % cs == 0 and pc % sps == 0 and pc % 2 == 0 and sps % nw == 0 and nrt % G == 0 \
        and G % hl == 0 and nrt // G * cs <= V31_SMS and _v31_smem((nt, cs, nw, st, sps, hl, f32), K, G) <= V31_SMEM_MAX


def _v31_instance(key, nt, plan, tune):
    _, _, hl, f32, _ = V31_LAYERS[key]
    return (nt, plan[0], tune[0], tune[1], tune[2], hl, f32)


def _v31_name(inst):
    nt, cs, nw, st, sps, hl, f32 = inst
    return f"bf16_v31<{nt},{cs},{nw},{st},{sps},{hl},{'true' if f32 else 'false'}>"


def _v31_candidates(key, nt):
    return [(p, t) for p in V31_PLANS[key] for t in V31_TUNES if _v31_valid(key, nt, p, t)]


V31_INSTANCES = sorted({_v31_instance(k, nt, p, t) for k in V31_LAYERS for nt in V31_NTS for (p, t) in _v31_candidates(k, nt)})


def _v31_launch(key, plan, tune, nt):
    """(grid CTAs, threads, dynamic _v31_smem bytes)."""
    rows, K, hl, f32, _ = V31_LAYERS[key]
    cs, G = plan
    return rows // 16 // G * cs, (tune[0] + 1) * 32, max(V31_SMEM_MIN, _v31_smem(_v31_instance(key, nt, plan, tune), K, G))


_V31 = {}  # "cubin", "lowered": {inst: name}; device -> (module, {inst: function})


def _v31_nts(bucket):
    return (1,) if bucket <= 8 else ((2,) if bucket <= 16 else (4,))


def _v31_insts():
    return V31_INSTANCES


def _v31_compile():
    """NVRTC (cuda.bindings, in-process) -> one sm_90a cubin with the needed instantiations. Host only."""
    if "cubin" not in _V31:
        from cuda.bindings import nvrtc
        insts = _v31_insts()
        err, prog = nvrtc.nvrtcCreateProgram(_V31_SRC.encode(), b"bf16_v31.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        for inst in insts:
            nvrtc.nvrtcAddNameExpression(prog, _v31_name(inst).encode())
        opts = [b"-arch=sm_90a", b"-std=c++17"]
        (res,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(log.decode(errors="ignore"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        lowered = {}
        for inst in insts:
            _, low = nvrtc.nvrtcGetLoweredName(prog, _v31_name(inst).encode())
            lowered[inst] = low.decode()
        _V31["cubin"], _V31["lowered"] = cubin, lowered
    return _V31["cubin"]


def _v31_load(dev):
    """CuPy functions on device dev (compile + load in prepare(), never inside graph capture)."""
    if dev not in _V31:
        import cupy as cp
        cubin = _v31_compile()
        with cp.cuda.Device(dev):
            mod = cp.cuda.function.Module()
            mod.load(cubin)
            fns = {}
            drv = cp.cuda.driver
            for inst, low in _V31["lowered"].items():
                f = mod.get_function(low)
                drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8),
                                     V31_SMEM_MAX)
                fns[inst] = f
        _V31[dev] = (mod, fns)
    return _V31[dev][1]


def _v31_repack(w):
    """[n, K] BF16 -> A-fragment order [n/16][K/32][s][lane][8 bf16] (lossless permutation, same bytes)."""
    n, K = w.shape
    t = w.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2)  # rt, h, g, p, t, s, e, q
    return t.permute(0, 3, 5, 2, 4, 6, 1, 7).contiguous()  # rt, p, s, g, t, e, h, q


def _v31_router_rows(hi, lo):
    """[256, K] hi, lo -> [512, K]: rowtile 2c = hi rows 16c.., rowtile 2c+1 = lo rows 16c.. (summed by HL = 2)."""
    n, K = hi.shape
    return torch.stack([hi.view(n // 16, 16, K), lo.view(n // 16, 16, K)], 1).reshape(2 * n, K)


def _v31(x, key, plan, tune, state):
    """out[m, n_out] = x . W^T on the v31 kernel (BF16 out; FP32 for the router)."""
    import cupy as cp
    import numpy as np
    m, K = x.shape
    rows, Kw, hl, f32, n_out = V31_LAYERS[key]
    nt = 1 if m <= 8 else (2 if m <= 16 else 4)
    assert K == Kw and m <= 8 * nt and x.is_contiguous() and x.data_ptr() % 16 == 0
    inst = _v31_instance(key, nt, plan, tune)
    grid, threads, shm = _v31_launch(key, plan, tune, nt)
    wr = state["v31_" + ("unpooler" if key == "unpooler_s" else key)]
    out = torch.empty((m, n_out), device=x.device, dtype=torch.float32 if f32 else torch.bfloat16)
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    f = _v31_load(dev)[inst]
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
        f((grid,), (threads,), (np.uint64(wr.data_ptr()), np.uint64(x.data_ptr()), np.int32(K),
                                np.uint64(out.data_ptr()), np.int32(m), np.int32(n_out), np.int32(K),
                                np.int32(plan[1])), shared_mem=shm)
    return out


def _v31_bucket(m):
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 2N]: hi columns, then lo
    if torch.cuda.is_available() and weights["qkv_gate"].is_cuda:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            state["v31_" + name] = _v31_repack(weights[name].contiguous())
        state["v31_router"] = _v31_repack(_v31_router_rows(hi, lo))
        _v31_load(weights["qkv_gate"].device.index or 0)  # NVRTC compile + module load now, not in capture
    try:
        _diag_bench(state)  # v31d: diagnostic only, untimed
    except Exception as e:  # never fail the run
        print("v31d diag failed:", repr(e)[:600], flush=True)
    return state


def _baseline(x, layer, n, state):
    """SGLang's own path (as in baseline.py), kept where it is faster."""
    if layer != "router":
        try:
            from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
        except Exception:  # local CPU testing only
            return x @ state[layer][:n].T
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.device.type == "cpu":  # local CPU testing only
        return x.float() @ (state["router_hi"].float() + state["router_lo"].float()).T
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m, k = x.shape
    if layer == "router" and m > TRITON_MAX_M:
        return _router_stacked(x, state)
    if m > TRITON_MAX_M:
        return _baseline(x, layer, n, state)
    key = "unpooler_s" if (layer == "unpooler" and n == 1024) else layer
    if CONFIGS[key] is None:
        return _baseline(x, layer, n, state)
    bn, bk, split, warps, stages = CONFIGS[key]
    bm = _bm(m)
    if layer == "router":
        w, lo = state["router_hi"], state["router_lo"]
    else:
        w, lo = state[layer], state[layer]
    x = x.contiguous()
    if split > 1:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm), split)
    _swapab_gemm[grid](x, w, lo, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                       BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_LO=(layer == "router"),
                       OUT_BF16=(split == 1), num_warps=warps, num_stages=stages)
    if layer != "router" and split > 1:
        out = out.to(x.dtype)
    return out


def _router_stacked(x, state):
    """Router above 32 tokens: SGLang's hi + lo pair as ONE cuBLAS GEMM against the stacked
    [hi; lo] weights (FP32 out), then hi-part + lo-part: the same two FP32-accumulated products
    and the same final FP32 add as SGLang's two torch.mm calls, in one GEMM launch
    (1.56-1.60x at 128 / 256 tokens in check c3f668686d154849)."""
    if x.device.type == "cpu":  # local CPU testing only
        both = x.float() @ state["router_hilo_t"].float()
    else:
        both = torch.mm(x, state["router_hilo_t"], out_dtype=torch.float32)
    n = both.shape[1] // 2
    return both[:, :n] + both[:, n:]


_DIAG = {}


def _diag_bench(state):
    """One-time untimed sweep: per (layer, m) the v6 path (kernel_fn as checked), SGLang's cuBLAS path and every
    v31 candidate, each captured in a CUDA graph and replayed with a cold L2 (a 256 MB buffer read first), median of
    15. Also: an empty-ish graph (one tiny kernel) as the fixed-cost floor, and max active clusters per cluster size."""
    import time
    if _DIAG or not torch.cuda.is_available():
        return
    _DIAG["done"] = True
    t0 = time.time()
    dev = state["qkv_gate"].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    tiny = torch.empty(16, device=dev)
    g = torch.Generator(device=dev).manual_seed(77)

    def timed(fn, reps=15):
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        ts = []
        for _ in range(reps):
            flush.view(torch.int32).sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    floor = timed(lambda: tiny.zero_())
    print(f"v31d diag: {torch.cuda.get_device_name()} SMs {torch.cuda.get_device_properties(dev).multi_processor_count}"
          f" floor(1 tiny kernel) {floor:.2f} us", flush=True)
    try:
        from cuda.bindings import driver as cu
        fns = _v31_load(dev.index or 0)
        seen = set()
        for key in V31_LAYERS:
            for (plan, tune) in _v31_candidates(key, 1)[:1]:
                if plan[0] in seen:
                    continue
                seen.add(plan[0])
                inst = _v31_instance(key, 1, plan, tune)
                grid, threads, shm = _v31_launch(key, plan, tune, 1)
                cfg = cu.CUlaunchConfig()
                cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
                cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = threads, 1, 1
                cfg.sharedMemBytes = shm
                r = cu.cuOccupancyMaxActiveClusters(cu.CUfunction(fns[inst].ptr), cfg)
                print(f"  CS {plan[0]} _v31_smem {shm >> 10} KB: max active clusters {r[1] if len(r) > 1 else r} "
                      f"(need {grid // plan[0]})", flush=True)
    except Exception as e:
        print("  occupancy query failed:", repr(e)[:200], flush=True)
    best = {}
    for key, (rows, K, hl, f32, n_out) in V31_LAYERS.items():
        layer = "unpooler" if key == "unpooler_s" else key
        wbytes = rows * K * 2
        for m in ((1, 8) if key == "unpooler_s" else (1, 8, 16, 32)):
            if time.time() - t0 > 240:
                print("(time guard)", flush=True)
                break
            x = torch.randn((m, K), generator=g, device=dev).to(torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": n_out, "k": K, "layer": layer}}
            if key == "router":
                ref = x.float() @ state["router"].float().t()
                t_cb = timed(lambda: _router_stacked(x, state))
            else:
                ref = _baseline(x, layer, n_out, state).float()
                t_cb = timed(lambda: _baseline(x, layer, n_out, state))
            t_v6 = timed(lambda: kernel_fn(data, state))
            res = []
            for plan, tune in _v31_candidates(key, 1 if m <= 8 else (2 if m <= 16 else 4)):
                try:
                    out = _v31(x, key, plan, tune, state).float()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                    t = timed(lambda: _v31(x, key, plan, tune, state))
                except Exception as e:
                    print(f"{key} m {m} {plan} {tune} FAILED {repr(e)[:120]}", flush=True)
                    continue
                res.append((t, plan, tune, err))
            res.sort()
            line = [f"{key} m {m}: v6 {t_v6:.2f} base {t_cb:.2f} | v31 " + "; ".join(
                f"{p}{tu} {t:.2f} ({wbytes / max(t - floor, 0.1) / 1e6:.2f} TB/s) e{err:.0e}" for t, p, tu, err in res[:3])]
            if res:
                best[(key, m)] = (res[0], t_v6)
            print("\n".join(line), flush=True)
    print("v31d summary (key m: best v31 / v6 us):", flush=True)
    for (key, m), ((t, p, tu, err), t6) in best.items():
        print(f"  {key}:{m} {t:.2f} / {t6:.2f} {list(p)} {list(tu)} e{err:.0e}", flush=True)
    print(f"v31d diag: done in {time.time() - t0:.0f} s", flush=True)
