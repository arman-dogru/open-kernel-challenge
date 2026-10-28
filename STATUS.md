# STATUS — updated Oct 28 2026, ~4:25 PM ET

Board (all provisional until the organisers' post-contest reruns):
- **Overall kernel board: rank 893 / ~2313** (was 1016 at 4:05 PM ET) (top-50 = serving priority; top-1000 = Meta interview pool).
- Prefill (`muse_prefill_layer`) has its own board and does not count toward overall — not attempted yet.

| problem | best submitted | rank | job | what it is |
|---|---|---|---|---|
| muse_bf16_proj | **1.098×** (95% 1.081–1.114) | 1142 | 735b0e423e784074 (v3_hybrid) | swap-AB Triton GEMM for o/pooler ≤32 tok; hi+lo router split-K ≤32 tok; cuBLAS elsewhere |
| muse_attention_decode | **1.075×** (1.059–1.091) | 946 / 958 | a08410ebc3b04bf7 (v5_best) | split-KV Triton decode + combine fused w/ norm·gate on sliding decode, full decode, sliding verify-16; FA3 elsewhere |
| muse_moe_w4a16 | **1.269×** (1.250–1.288) | 861 | 1b5c613200e84954 (v9_postnorm_fused) | SGLang Marlin, moe_block_size re-tuned at 128/256 tokens; SGLang's fused post-expert norm+sum Triton kernel launched directly below 32 tokens (its wrapper falls back to ~10 PyTorch kernels there): decode-1 1.48×, verify-1 1.27× |
| muse_decode_layer | **1.180×** (1.162–1.198) | 765 | 9cb6d6a409ff4cd6 (v4_postnorm_fused) | baseline + the swaps above (split-KV attention, Triton o/pooler/router, Marlin block sizes, direct post-expert norm kernel) |
| muse_prefill_layer | — | — | — | not started |

Reference points (per-problem speedup at rank 50 / 300 / 500 / 900, Oct 28): bf16 1.55/1.37/1.26/1.17,
attention 2.61/1.94/1.62/1.13, MoE 2.36/1.95/1.72/1.24, decode_layer 2.14/1.73/1.46/—.

## Quotas used today (UTC day Oct 28; resets 8 PM ET)
| problem | submits used | checks used |
|---|---|---|
| bf16_proj | 1 / 10 | 5 / 20 |
| attention_decode | 2 / 10 | 8 / 20 (2 were Triton compile failures — now caught offline by tools/compile_sm90.py) |
| moe_w4a16 | 2 / 10 | 9 / 20 |
| decode_layer | 5 / 10 | 6 / 20 |
| prefill_layer | 0 / 10 | 0 / 20 |

## Kaggle GPU (cap 15 h for the Kaggle week ending Oct 31 00:00 UTC, shared with Developper)
- Used by this agent: 1 probe session (~3 min, ≈0.05 h) on 2×T4 — confirmed torch 2.11 / Triton 3.6, bf16 `tl.dot` works on T4.
- Verdict: low value. T4 is sm_75 (no FP8, no wgmma/TMA, no H200 timing). CPU `TRITON_INTERPRET=1` tests +
  offline sm_90 compile (`tools/compile_sm90.py`) cover logic and compile errors without spending GPU hours.
  Reserve Kaggle for a future CUDA `load_inline` kernel that needs a real nvcc+GPU correctness run.

## What we learned (H200 check runs; profiles in `<problem>/runs/profiles/`)
- Below 32 tokens SGLang's `fused_post_expert_norm_weighted_sum` runs a PyTorch fallback (~10 kernels,
  ~25 µs); launching its own Triton kernel at every token count was the cheapest win (MoE 1.10→1.27×,
  decode_layer 1.11→1.18×).
- decode_layer v5 (check 174d8047c762425e): a true-FP32 Triton `ieee` split-K GEMV for the ≤8-token router
  was *slower* than cuBLAS's FP32 gemv (decode-1 137→155 µs) — not submitted.
- bf16_proj: Triton swap-AB wins only where each CTA has a short K range (o: K=1024) or BN≥32; qkv_gate
  (BN=16) and split-K + atomics (+memset +cast kernels) lose to cuBLAS. At m≥128 cuBLAS wins everywhere,
  incl. a Triton router (35 µs vs 16 µs). Router ≤32 tokens: 1.3–2.6×.
- MoE: a Triton NVFP4 dequant-in-registers grouped GEMM (runs/wip_triton_moe.py, v1–v4) is correct but
  2–4× slower than Marlin — ALU-bound on per-element E2M1/E4M3 decode. Needs packed (2–4 values/instr)
  PTX dequant (`tl.inline_asm_elementwise` or CUDA `load_inline`, Marlin-style LOP3/prmt) to compete.
- Attention: page_size=1 FA3 on full layers is ~3× off the floor (370 µs vs 126 µs at b=32). Our split-KV
  kernel beats FA3 by 1.07× on full decode and 1.25–1.35× on sliding decode; verify-8 full is still worse
  (64–128-row blocks), so FA3 is kept there.
