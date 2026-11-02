# okc: problem=muse_bf16_proj
"""v30: v6 plus a weight-streaming CUDA kernel (tools/bf16_v30_stream.cu: W repacked into mma.sync A-fragment
order, 512-byte LDG.128 streams with bulk L2 prefetch, split-K reduced in shared memory and across a thread-block
cluster through DSMEM, no atomics, persistent grid) on the (layer, m) pairs in V30_ROUTE, picked from the v30d
diagnostic's per-shape timings vs v6 / cuBLAS. Router unchanged (v6's stacked hi/lo). tools/build_bf16_v30.py.

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



# ---------------- v30: weight-streaming CUDA kernel (tools/bf16_v30_stream.cu, tools/bf16_v30_cfg.py) ----------------
_V30_SRC = r"""
// bf16 v30: weight-streaming skinny GEMM out[m, n] = x[m, K] . W[n, K]^T for m <= 32 (BF16 in, FP32 accumulate,
// BF16 out), sm_90a, mma.sync m16n8k16 swap-AB (W rows = the MMA's M side, tokens = its N side).
// - W is repacked once in prepare() into A-fragment order: [rowtile (16 rows)][pair (32 k)][step s][lane][4 x b32],
//   so a warp streams its (rowtile, k-range) as consecutive 512-byte LDG.128s (L1 no-allocate, L2 evict-first) and
//   each 16-byte load is one MMA's whole A fragment. k is permuted inside each 32-wide pair (lane t, step s holds
//   k = 8t + 4s + 0..3) identically for W and x, so x's B fragments are one 16-byte load per (token tile, pair).
// - CTA = RG x WK warps: RG rowtile groups (RT rowtiles per warp) x WK warps along k; CS CTAs per cluster split k
//   further. Each warp loads GP pairs (GP * RT KB) per chunk; PFD chunks ahead are bulk-prefetched into L2
//   (cp.async.bulk.prefetch.L2), which is what keeps enough bytes in flight (ptxas serializes the register loads).
// - Split-K reduction without atomics: in-CTA through shared memory (fixed order), then across the cluster through
//   distributed shared memory into rank 0, which writes BF16. Deterministic; no workspace.
// - Persistent: grid = min(units, cap) clusters; a cluster loops over units (unit = RG * RT rowtiles).
typedef unsigned int u32;
typedef unsigned short u16;

__device__ __forceinline__ uint4 ldw(const uint4* p, unsigned long long pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void pf_l2(const void* p, u32 bytes) {  // bulk L2 prefetch (fire and forget)
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint4 ldx(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
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
__device__ __forceinline__ u32 cluster_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_dsmem(const float* p, u32 rank) {
  u32 a = (u32)__cvta_generic_to_shared(p), ra;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}

template <int RT, int NT, int WK, int RG, int CS, int GP, int PFD = 2>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(32 * WK * RG)
bf16_stream(const uint4* __restrict__ W, const u16* __restrict__ x, int ldx_, u16* __restrict__ out, int m, int n,
            int K, int units) {
  constexpr int E = RG * RT * NT * 128;  // FP32 partial outputs per CTA per unit
  extern __shared__ float red[];         // [WK][E]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rg = warp / WK, ks = warp % WK;
  const int g = lane >> 2, t = lane & 3;
  const u32 rank = CS > 1 ? cluster_rank() : 0u;
  const int cid = blockIdx.x / CS, ncl = gridDim.x / CS;
  const int KP = K >> 5;                 // pairs per rowtile
  const int PW = KP / (WK * CS);         // pairs per warp
  const int p0 = ((int)rank * WK + ks) * PW;
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  for (int u = cid; u < units; u += ncl) {
    const int rt0 = (u * RG + rg) * RT;
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][j][c] = 0.f;
    const uint4* wp[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wp[r] = W + ((size_t)(rt0 + r) * KP + p0) * 64 + lane;
    // ptxas interleaves the A loads with their MMAs (~2 in flight per warp), so the DRAM latency is hidden by
    // bulk L2 prefetches of the warp's next chunks instead: PFD chunks (GP KB per rowtile each) ahead.
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
        for (int q = 0; q < PFD && q * GP < PW; ++q) pf_l2(wp[r] - lane + (size_t)q * GP * 64, GP * 1024);
    }
    for (int c = 0; c < PW; c += GP) {
      if (lane == 0 && c + PFD * GP < PW) {
#pragma unroll
        for (int r = 0; r < RT; ++r) pf_l2(wp[r] - lane + (size_t)(c + PFD * GP) * 64, GP * 1024);
      }
      uint4 a[GP][RT][2];
#pragma unroll
      for (int i = 0; i < GP; ++i)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          a[i][r][0] = ldw(wp[r] + (size_t)(c + i) * 64, pol);
          a[i][r][1] = ldw(wp[r] + (size_t)(c + i) * 64 + 32, pol);
        }
#pragma unroll
      for (int i = 0; i < GP; ++i) {
        const int kb = ((p0 + c + i) << 5) + 8 * t;
        uint4 b[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int col = 8 * j + g;
          b[j] = col < m ? ldx(reinterpret_cast<const uint4*>(x + (size_t)col * ldx_ + kb)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            mma(acc[r][j], a[i][r][0].x, a[i][r][0].y, a[i][r][0].z, a[i][r][0].w, b[j].x, b[j].y);
            mma(acc[r][j], a[i][r][1].x, a[i][r][1].y, a[i][r][1].z, a[i][r][1].w, b[j].z, b[j].w);
          }
      }
    }
    // in-CTA split-K reduction (fixed order over ks)
    float* my = red + ks * E + rg * RT * NT * 128;
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) my[((r * NT + j) * 4 + c) * 32 + lane] = acc[r][j][c];
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < WK; ++w) s += red[w * E + e];
      red[e] = s;
    }
    if (CS > 1) cluster_sync(); else __syncthreads();
    if (rank == 0) {
      for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float s = red[e];
#pragma unroll
        for (int q = 1; q < CS; ++q) s += ld_dsmem(red + e, (u32)q);
        const int ln = e & 31, c = (e >> 5) & 3, rj = e >> 7, j = rj % NT, R = rj / NT;
        const int row = (u * RG * RT + R) * 16 + (ln >> 2) + 8 * (c >> 1);
        const int col = 8 * j + 2 * (ln & 3) + (c & 1);
        if (col < m && row < n) out[(size_t)col * n + row] = f2bf(s);
      }
    }
    if (CS > 1) cluster_sync(); else __syncthreads();
  }
}
"""
# tools/bf16_v30_cfg.py (body; names prefixed _v30 / V30_)

V30_CONFIGS = [(1, 8, 1, 1, 8), (1, 16, 1, 1, 8), (1, 8, 1, 1, 16), (2, 8, 1, 1, 8), (1, 4, 2, 1, 8), (1, 8, 1, 2, 8),
           (1, 8, 1, 4, 4), (1, 8, 1, 2, 4), (2, 4, 2, 1, 8), (1, 4, 4, 1, 4), (1, 8, 2, 1, 4), (2, 8, 1, 2, 8),
           (1, 4, 2, 2, 8), (2, 4, 1, 2, 8)]
V30_NTS = (1, 2, 4)
V30_INSTANCES = [(rt, nt, wk, rg, cs, gp) for (rt, wk, rg, cs, gp) in V30_CONFIGS for nt in V30_NTS]
# layer -> (n rows used, K)
V30_LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "unpooler_s": (1024, 4096)}


def _v30_name(inst):
    rt, nt, wk, rg, cs, gp = inst
    return f"bf16_stream<{rt}, {nt}, {wk}, {rg}, {cs}, {gp}>"


def _v30_valid(cfg, n, K, nt):
    rt, wk, rg, cs, gp = cfg
    rows = n // 16
    if n % 16 or rows % (rg * rt) or K % 32:
        return False
    kp = K // 32
    if kp % (wk * cs):
        return False
    pw = kp // (wk * cs)
    if pw < gp or pw % gp:
        return False
    return wk * rg * 32 <= 512 and _v30_smem(cfg, nt) <= 96 * 1024


def _v30_smem(cfg, nt):
    rt, wk, rg, cs, gp = cfg
    return wk * rg * rt * nt * 128 * 4


def _v30_grid(cfg, n, cap=264):
    """CTAs: one cluster per unit up to cap CTAs (persistent: clusters loop over units)."""
    rt, wk, rg, cs, gp = cfg
    units = (n // 16) // (rg * rt)
    return min(units, max(1, cap // cs)) * cs, units


_V30 = {}  # "cubin", "lowered": {inst: name}; device -> {inst: function}


def _v30_compile():
    """NVRTC (cuda.bindings, in-process) -> one sm_90a cubin with every instantiation. Host only."""
    if "cubin" not in _V30:
        from cuda.bindings import nvrtc
        err, prog = nvrtc.nvrtcCreateProgram(_V30_SRC.encode(), b"bf16_stream.cu", 0, [], [])
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
        for inst in V30_INSTANCES:
            nvrtc.nvrtcAddNameExpression(prog, _v30_name(inst).encode())
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
        for inst in V30_INSTANCES:
            _, low = nvrtc.nvrtcGetLoweredName(prog, _v30_name(inst).encode())
            lowered[inst] = low.decode()
        _V30["cubin"], _V30["lowered"] = cubin, lowered
    return _V30["cubin"]


def _v30_load(dev):
    """CuPy functions on device dev (compile + load in prepare(), never inside graph capture)."""
    if dev not in _V30:
        import cupy as cp
        cubin = _v30_compile()
        with cp.cuda.Device(dev):
            mod = cp.cuda.function.Module()
            mod.load(cubin)
            fns = {}
            drv = cp.cuda.driver
            for inst, low in _V30["lowered"].items():
                f = mod.get_function(low)
                rt, nt, wk, rg, cs, gp = inst
                drv.funcSetAttribute(f.ptr, getattr(drv, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", 8),
                                     _v30_smem((rt, wk, rg, cs, gp), nt))
                fns[inst] = f
        _V30[dev] = (mod, fns)
    return _V30[dev][1]


def _v30_repack(w):
    """[n, K] BF16 -> A-fragment order [n/16][K/32][s][lane][8 bf16] (lossless permutation, same bytes)."""
    n, K = w.shape
    t = w.reshape(n // 16, 2, 8, K // 32, 4, 2, 2, 2)  # rt, h, g, p, t, s, e, q
    return t.permute(0, 3, 5, 2, 4, 6, 1, 7).contiguous()  # rt, p, s, g, t, e, h, q


def _v30(x, key, n, cfg, state):
    """out[m, n] (BF16) = x . W[:n]^T on the v30 kernel with cfg = (RT, WK, RG, CS, GP)."""
    import cupy as cp
    import numpy as np
    m, K = x.shape
    nt = (m + 7) // 8
    rt, wk, rg, cs, gp = cfg
    wr = state["v30_" + ("unpooler" if key == "unpooler_s" else key)]
    assert x.is_contiguous() and x.data_ptr() % 16 == 0 and K % 8 == 0
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    nblk, units = _v30_grid(cfg, n)
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    f = _v30_load(dev)[(rt, nt, wk, rg, cs, gp)]
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
        f((nblk,), (32 * wk * rg,), (np.uint64(wr.data_ptr()), np.uint64(x.data_ptr()), np.int32(K),
                                     np.uint64(out.data_ptr()), np.int32(m), np.int32(n), np.int32(K),
                                     np.int32(units)), shared_mem=_v30_smem(cfg, nt))
    return out


def _v30_bucket(m):
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))


# (layer key, m bucket 1 / 8 / 16 / 32) -> v30 config, from the v30d diagnostic
V30_ROUTE = {
    ('unpooler_s', 1): (1, 16, 1, 1, 8),
    ('unpooler_s', 8): (1, 16, 1, 1, 8),
}

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
            state["v30_" + name] = _v30_repack(weights[name].contiguous())
        _v30_load(weights["qkv_gate"].device.index or 0)  # NVRTC compile + module load now, not in capture
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
    v30cfg = V30_ROUTE.get((key, _v30_bucket(m)))
    if v30cfg is not None and x.is_cuda:
        return _v30(x.contiguous(), key, n, v30cfg, state)
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
