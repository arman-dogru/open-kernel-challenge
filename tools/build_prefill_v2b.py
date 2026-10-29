"""Builds muse_prefill_layer/runs/v2b_dequant_grouped.py from v1_router_fused.py (v2 plus the v2b fixes in its docstring) (exact replacements, asserted once).
Run at HEAD; the committed file must be byte-identical to its output."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s = (ROOT / "muse_prefill_layer/runs/v1_router_fused.py").read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''"""prefill v1 = SGLang's layer''', '''"""prefill v2b = v2 with its two slow steps fixed. In check b007869e4f7e48ef the grouped GEMMs took 4.5 ms where
Marlin took 11.7 ms, but the dequant took 3.2 ms and the scatter back to slot order 2.4 ms:
  - the dequant runs one program per weight row (2048 bytes), so the row, expert and scale indices are scalars or
    shifts (v2 did three int64 div / mod per byte), and it stores lo / hi interleaved as one contiguous BF16 row.
  - the outputs go back to (token, slot) order as a gather through the inverse permutation (index_select) instead
    of an index_put scatter.

prefill v2 = v1 with the routed experts at >= 8192 tokens (the first-turn 16K-token chunks) run as BF16 tensor-core
grouped GEMMs instead of Marlin W4A16. Marlin ran those at ~145 TFLOP/s: 9.6-11.8 ms of 16-17 ms per shape in
check e09cfcf550ab476e. Each call:
  1. dequantizes the local NVFP4 experts (w13, w2: e2m1 * e4m3 block scale * FP32 tensor scale, one rounding to
     BF16) with a Triton kernel into a per-call buffer. The stored weights stay NVFP4; nothing derived from them is
     kept between calls.
  2. stable-sorts this GPU's routes by local expert (non-local routes last) and gathers their pre-expert-normed rows.
  3. GEMM-1 as torch._grouped_mm with per-expert row offsets, then SGLang's silu_and_mul (non-local rows skipped by
     expert id -1), then GEMM-2 likewise.
  4. puts the outputs back in (token, slot) order for SGLang's own post-expert-norm weighted sum, as the Marlin
     path does.
Below 8192 tokens (753 / 3012-token turns and all 5 tests) the experts stay on Marlin; there the per-call dequant
(~3.2 GB of BF16 written) would cost more than Marlin does.

v1 = SGLang's layer''')
rep('''from torch import nn
''', '''from torch import nn
import triton
import triton.language as tl
''')
rep('''def _experts(x, topk_weights, topk_ids, state):''', '''# Token counts at or above this run the experts as dequantized BF16 grouped GEMMs (v2); below, Marlin.
GROUPED_MIN_TOKENS = 8192


@triton.jit
def _nvfp4_dequant_kernel(q_ptr, sf_ptr, gs_ptr, out_ptr, rows_per_expert, ROW_BYTES: tl.constexpr):
    # uint8 [E, N, K/2] (element 2i in the low nibble) with e4m3 [E, N, K/16] and FP32 [E] -> BF16 [E, N, K];
    # one program per weight row
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, ROW_BYTES)
    b = tl.load(q_ptr + row * ROW_BYTES + c).to(tl.int32)
    sfb = tl.load(sf_ptr + row * (ROW_BYTES // 8) + c // 8)  # e4m3 scale bytes as uint8
    scale = sfb.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(gs_ptr + row // rows_per_expert)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    mlo = lo & 7
    mhi = hi & 7
    # E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 for codes 0..7
    vlo = tl.where(mlo < 4, mlo.to(tl.float32) * 0.5, tl.where(mlo < 6, (mlo - 2).to(tl.float32),
                   tl.where(mlo == 6, 4.0, 6.0)))
    vhi = tl.where(mhi < 4, mhi.to(tl.float32) * 0.5, tl.where(mhi < 6, (mhi - 2).to(tl.float32),
                   tl.where(mhi == 6, 4.0, 6.0)))
    vlo = tl.where((lo & 8) != 0, -vlo, vlo) * scale
    vhi = tl.where((hi & 8) != 0, -vhi, vhi) * scale
    v = tl.reshape(tl.join(vlo, vhi), (2 * ROW_BYTES,))  # element 2i = low nibble, 2i + 1 = high
    tl.store(out_ptr + row * (2 * ROW_BYTES) + tl.arange(0, 2 * ROW_BYTES), v.to(tl.bfloat16))


def _dequant(q, sf, gs):
    e, n, kb = q.shape
    out = torch.empty((e, n, 2 * kb), device=q.device, dtype=torch.bfloat16)
    _nvfp4_dequant_kernel[(e * n,)](q, sf.view(torch.uint8), gs, out, n, ROW_BYTES=kb, num_warps=8)
    return out


def _experts_grouped(x, topk_weights, topk_ids, state):
    """The routed experts as BF16 grouped GEMMs on per-call dequantized weights (see the docstring)."""
    bank, raw = state["bank"], state["raw"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    local_ids = bank["expert_map"][topk_ids].contiguous()  # [T, 8], -1 where not on this GPU
    topk_weights = topk_weights.to(torch.float32).contiguous()
    flat = local_ids.view(-1)
    key = torch.where(flat >= 0, flat, LOCAL_EXPERTS)
    skey, perm = torch.sort(key, stable=True)
    counts = torch.zeros((LOCAL_EXPERTS + 1,), device=x.device, dtype=torch.int32)
    counts.scatter_add_(0, key, torch.ones_like(key, dtype=torch.int32))
    offs = torch.cumsum(counts[:LOCAL_EXPERTS], 0, dtype=torch.int32)
    w13 = _dequant(raw["w13"], raw["w13_sf"], raw["w13_gs"])  # [32, 2I, H]
    xs = x[perm // TOP_K]  # [T*8, H] in expert order (non-local routes last, unused)
    h1 = torch._grouped_mm(xs, w13.transpose(1, 2), offs=offs)  # [T*8, 2I]; rows past offs[-1] unwritten
    del w13
    sorted_eid = torch.where(skey < LOCAL_EXPERTS, skey, -1).to(local_ids.dtype)
    h2 = torch.empty((h1.shape[0], size_n), device=x.device, dtype=x.dtype)
    silu_and_mul(h1, h2, expert_ids=sorted_eid, expert_step=1)
    del h1
    w2 = _dequant(raw["w2"], raw["w2_sf"], raw["w2_gs"])  # [32, H, I]
    ys = torch._grouped_mm(h2, w2.transpose(1, 2), offs=offs)  # [T*8, H]
    del w2
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=x.device, dtype=perm.dtype)
    y = ys.index_select(0, inv)  # back to (token, slot) order; non-local slots are skipped by the finalize (id -1)
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, TOP_K, size_k), topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=local_ids)


def _experts(x, topk_weights, topk_ids, state):''')
rep('''        "bank": _marlin_bank(weights),''', '''        "bank": _marlin_bank(weights),
        # the NVFP4 experts as given (v2's grouped path dequantizes them per call; nothing else is stored)
        "raw": {name: weights[name].contiguous() for name in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")},''')
rep('''    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)''',
'''    experts_fn = _experts_grouped if tokens >= GROUPED_MIN_TOKENS else _experts
    expert_output = experts_fn(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)''')
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "muse_prefill_layer/runs/v2b_dequant_grouped.py"
dst.write_text(s)
