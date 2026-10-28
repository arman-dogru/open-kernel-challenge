# okc: problem=muse_moe_w4a16
"""Routed experts (NVFP4 W4A16) with two Triton grouped GEMMs, swap-AB for small token counts.

- Routing: SGLang's moe_align_block_size over the 32 local experts (+1 bucket for the routes
  that go to other GPUs, which is skipped), the same graph-safe op the baseline uses.
- GEMM 1: per (expert block, 64-wide slice of I), gate and up rows are dequantized in registers
  (E2M1 codes and E4M3 block scales by exact bit placement into FP16, product rounded to BF16,
  which is exact) and fed to BF16 tensor-core dots with FP32 accumulation; the epilogue applies
  the global scale and silu(gate) * up, storing the BF16 intermediate (as SGLang does).
- GEMM 2: the same for the down projection, BF16 output per route.
- SGLang's Triton post-expert norm + weighted sum combines the routes.
Weights stay NVFP4 (untouched); no precision is dropped."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2]
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H]


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


POST_NORM_EPS = 1e-8
# 2^14 (E2M1 placed in FP16 bits) * 2^8 (E4M3 placed in FP16 bits)
DEQ_FIX = float(2 ** 22)


@triton.jit
def _deq_tile(wq, sf, BN: tl.constexpr, BK: tl.constexpr, DT: tl.constexpr):
    """wq uint8 [BN, BK//2] packed E2M1 (low nibble first), sf uint8 [BN, BK//16] E4M3 bits
    -> DT [BN, BK] = e2m1 * e4m3 / 2^22 (exact)."""
    lo = (wq & 0xF).to(tl.int16)
    hi = ((wq >> 4) & 0xF).to(tl.int16)
    c = tl.reshape(tl.join(lo, hi), (BN, BK))
    v = (((c & 7) << 9) | ((c & 8) << 12)).to(tl.float16, bitcast=True).to(tl.float32)
    s16 = sf.to(tl.int16)
    s = (((s16 & 0x7F) << 7) | ((s16 & 0x80) << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    w = tl.reshape(v, (BN, BK // 16, 16)) * tl.reshape(s, (BN, BK // 16, 1))
    return tl.reshape(w, (BN, BK)).to(DT)


@triton.jit
def _moe_gemm(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, eids_ptr, npad_ptr,
              numel, max_blocks, local_start,
              N, K, A_ROW_DIV: tl.constexpr, GATED: tl.constexpr, N_LOCAL: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NUM_PM: tl.constexpr):
    """out[route, n] for route in sorted blocks. Weights per expert: [ROWS, K/2] packed, rows
    [0, N) (and gate/up: up rows [N, 2N)). A rows: a[route // A_ROW_DIV]."""
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    npad = tl.load(npad_ptr)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_k2 = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 16)
    DT = a_ptr.dtype.element_ty
    if GATED:
        rows_per_e = 2 * N
    else:
        rows_per_e = N
    for blk in range(pid_p, max_blocks, NUM_PM):
        e = tl.load(eids_ptr + blk)
        if (blk * BM < npad) & (e >= 0) & (e < N_LOCAL):
            routes = tl.load(sorted_ptr + blk * BM + tl.arange(0, BM))
            valid = routes < numel
            arow = tl.where(valid, routes // A_ROW_DIV, 0)
            eg = (e + local_start).to(tl.int64)
            wbase = w_ptr + eg * rows_per_e * (K // 2)
            sbase = sf_ptr + eg * rows_per_e * (K // 16)
            a_ptrs = a_ptr + arow[None, :].to(tl.int64) * K + offs_k[:, None]
            wg_ptrs = wbase + offs_n[:, None] * (K // 2) + offs_k2[None, :]
            sg_ptrs = sbase + offs_n[:, None] * (K // 16) + offs_ks[None, :]
            acc_g = tl.zeros((BN, BM), dtype=tl.float32)
            acc_u = tl.zeros((BN, BM), dtype=tl.float32)
            for kk in range(0, K, BK):
                at = tl.load(a_ptrs + kk, mask=valid[None, :], other=0.0)
                wg = _deq_tile(tl.load(wg_ptrs + kk // 2), tl.load(sg_ptrs + kk // 16), BN, BK, DT)
                acc_g = tl.dot(wg, at, acc_g)
                if GATED:
                    wu = _deq_tile(tl.load(wg_ptrs + N * (K // 2) + kk // 2),
                                   tl.load(sg_ptrs + N * (K // 16) + kk // 16), BN, BK, DT)
                    acc_u = tl.dot(wu, at, acc_u)
            gs = tl.load(gs_ptr + eg) * 4194304.0
            if GATED:
                g = acc_g * gs
                u = acc_u * gs
                res = g * tl.sigmoid(g) * u
            else:
                res = acc_g * gs
            o_ptrs = out_ptr + routes[None, :].to(tl.int64) * N + offs_n[:, None]
            tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty), mask=valid[None, :])


# (BN, BK, num_warps, num_stages) for the two GEMMs
CFG1 = (64, 128, 4, 3)
CFG2 = (64, 128, 4, 3)
BM = 16


def _align(local_ids, block_m, num_experts):
    try:
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    except Exception:  # local CPU testing only
        return _align_torch(local_ids, block_m, num_experts)
    return moe_align_block_size(local_ids, block_m, num_experts)


def _align_torch(ids, block_m, num_experts):
    flat = ids.flatten().long()
    numel = flat.numel()
    maxp = numel + num_experts * (block_m - 1)
    sorted_ids = torch.full((maxp,), numel, dtype=torch.int32)
    eids = torch.full((maxp // block_m,), -1, dtype=torch.int32)
    pos = 0
    for e in range(num_experts):
        idx = (flat == e).nonzero().flatten()
        if idx.numel() == 0:
            continue
        nb = -(-idx.numel() // block_m)
        sorted_ids[pos:pos + idx.numel()] = idx.int()
        eids[pos // block_m: pos // block_m + nb] = e
        pos += nb * block_m
    return sorted_ids, eids, torch.tensor([pos], dtype=torch.int32)


def _post_norm_sum(y, topk_weights, norm_w, slot_ids):
    try:
        from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
            fused_post_expert_norm_weighted_sum,
        )
    except Exception:  # local CPU testing only
        yy = (y.float() * norm_w.float()).to(torch.bfloat16).float()
        yy = yy * torch.rsqrt(yy.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        w = topk_weights * (slot_ids >= 0)
        yy = torch.where((slot_ids >= 0)[..., None], yy, 0.0)
        return (yy * w[..., None]).sum(1).to(torch.bfloat16)
    return fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=norm_w, eps=POST_NORM_EPS,
                                               weight_first=True, slot_expert_ids=slot_ids)


def prepare(weights):
    return {
        "w13": weights["w13"].contiguous(),
        "w13_sf": weights["w13_sf"].contiguous().view(torch.uint8),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": weights["w2"].contiguous(),
        "w2_sf": weights["w2_sf"].contiguous().view(torch.uint8),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    bs, H = x.shape
    I = cfg["d_expert"]
    top_k = cfg["top_k"]
    start, count = cfg["local_start"], cfg["local_count"]
    ids = data["topk_ids"].to(torch.int32) - start
    is_local = (ids >= 0) & (ids < count)
    slot_ids = torch.where(is_local, ids, -1).contiguous()
    bucket_ids = torch.where(is_local, ids, count).contiguous()  # non-local -> bucket `count`
    n_buckets = 64 if count < 64 else count + 1
    sorted_ids, eids, npad = _align(bucket_ids, BM, n_buckets)
    numel = bs * top_k
    max_blocks = eids.numel()
    num_pm = min(max_blocks, max(8, min(32, numel // 4)))

    inter = torch.empty((numel, I), device=x.device, dtype=x.dtype)
    bn, bk, warps, stages = CFG1
    _moe_gemm[(I // bn, num_pm)](
        x, state["w13"], state["w13_sf"], state["w13_gs"], inter, sorted_ids, eids, npad,
        numel, max_blocks, start, I, H, A_ROW_DIV=top_k, GATED=True, N_LOCAL=count,
        BM=BM, BN=bn, BK=bk, NUM_PM=num_pm, num_warps=warps, num_stages=stages)
    y = torch.empty((numel, H), device=x.device, dtype=x.dtype)
    bn, bk, warps, stages = CFG2
    _moe_gemm[(H // bn, num_pm)](
        inter, state["w2"], state["w2_sf"], state["w2_gs"], y, sorted_ids, eids, npad,
        numel, max_blocks, start, H, I, A_ROW_DIV=1, GATED=False, N_LOCAL=count,
        BM=BM, BN=bn, BK=bk, NUM_PM=num_pm, num_warps=warps, num_stages=stages)
    return _post_norm_sum(y.view(bs, top_k, H), data["topk_weights"].float().contiguous(),
                          state["norm_w"], slot_ids)
