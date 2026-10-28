# STATUS — updated Oct 28 2026, ~4:40 PM ET

**Rules acceptance: 2026-10-28 ~6 PM ET, rules re-accepted by Arman personally via `okc login` (on his own
Mac); the box's earlier agent-accepted session was logged out and revoked. Submits from here use his token.**

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

## Quotas used today (UTC day Oct 28; resets 8 PM ET), from `okc problems`
| problem | submits used | checks used |
|---|---|---|
| bf16_proj | 1 / 10 | 5 / 20 |
| attention_decode | 2 / 10 | 10 / 20 (2 were Triton compile failures) |
| moe_w4a16 | 2 / 10 | 9 / 20 |
| decode_layer | 4 / 10 | 6 / 20 |
| prefill_layer | 0 / 10 | 0 / 20 |

## Kaggle GPU (cap 15 h for the Kaggle week ending Oct 31 00:00 UTC, shared with Developper)
- Used by this agent: 1 probe session (~3 min, ≈0.05 h) on 2×T4 — confirmed torch 2.11 / Triton 3.6, bf16 `tl.dot` works on T4.
- Verdict: low value. T4 is sm_75 (no FP8, no wgmma/TMA, no H200 timing). CPU `TRITON_INTERPRET=1` tests +
  offline sm_90 compile (`tools/compile_sm90.py`) cover logic and compile errors without spending GPU hours.
  Reserve Kaggle for a future CUDA `load_inline` kernel that needs a real nvcc+GPU correctness run.

## Queued (committed before any send)
- `muse_attention_decode/runs/v9_regime.py`: Peer Review M1 fix. Paths chosen by regime, not exact batch:
  sliding decode → split-KV head; full decode b=1 → rows 4/SM; full decode b≥2 → rows 16/SM + V-load
  hoist; every verify block and untuned head layouts → FA3. Same kernels as check bb656112f11e4320
  (v8_rows_occ, **~1.12×**, 3,355 µs vs 3,504.5 µs submitted) except sliding verify b16 now FA3 (1.08→1.0×
  on that shape, ~2% of one load). CPU tests pass (incl. b=2 full decode through the new path).
  Plan: one check, then submit if ≥ 1.10×.
- `muse_decode_layer/runs/v6_attn_regime.py`: v4 (submitted, 1.180×) + the v9_regime attention path and
  kernels (verbatim), replacing the exact-shape SPLITKV_ON table. Plan: check, submit if > 1.180×.

## Audit r1 follow-ups
- M1 (attention exact-batch routing): fixed in v9_regime (above); decode_layer v6 also by regime.
- M2 (who ran `okc login`): resolved — Arman re-accepted personally ~6 PM ET Oct 28; box session revoked.
- m1: job ledger below; m2/m3: new rule — every runs/ file is committed before it is sent and never
  edited after (new vN name instead); `LICENSE` (Apache-2.0) added.

## What we learned (H200 check runs; profiles in `<problem>/runs/profiles/`)
- Attention full decode: split-KV "rows" (both KV heads per program, whole 128-byte cache rows) with
  ~16 programs/SM and the V gather issued before the QK math: 160 / 296 µs at b16 / b32 vs 178 / 349 µs
  for the old 4/SM "head" config (FA3: 193 / 375). The combine needs 4 warps at ≥64 splits (7→~2 µs at
  b1). L=8 verify through split-KV remains far slower than FA3 (rows 881 µs, head 239–572 µs vs 193–376).
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

## Job ledger (every okc job; times ET; sha256 = first 16 hex of the runs/ file as committed)

Caveat (audit m2): files before this ledger were not hashed at send time; `04269b7477f74417`'s
runs/v4_postnorm_fused.py was rewritten 5 s after that check finished (docstring only, by a script);
the board job `9cb6d6a409ff4cd6` was sent after that rewrite, so it matches the committed bytes.

| time | job | problem | kind | file | sha256[:16] | status | time |
|---|---|---|---|---|---|---|---|
| Oct 28 14:52 | 85916cc3c560491c | muse_bf16_proj | check | v0_baseline.py | 9dd0eb54223e8474 | passed | 4,810.0 µs |
| Oct 28 14:52 | 89942ec8264a428f | muse_decode_layer | check | v0_baseline.py | 2bddf929e2ba027d | passed | 22.59 ms |
| Oct 28 14:52 | 8d3eb1174f1f400e | muse_moe_w4a16 | check | v0_baseline.py | 66abb265321ae011 | passed | 13.46 ms |
| Oct 28 14:52 | 9e417014ab3d4f5e | muse_attention_decode | check | v0_baseline.py | 4710dccc7391b77d | passed | 3,771.0 µs |
| Oct 28 15:00 | 6328a6919e404d03 | muse_bf16_proj | check | v1_swapab.py | 5d4d9a755481c8f8 | passed | 6,895.2 µs |
| Oct 28 15:02 | 135ef58e128d4101 | muse_moe_w4a16 | check | v1_triton.py | 086a31e6387e053d | passed | 26.55 ms |
| Oct 28 15:03 | e1b313dd0f304c90 | muse_bf16_proj | check | v2_splitk.py | bee59389a3208467 | passed | 5,131.7 µs |
| Oct 28 15:05 | 6ca23a961a2e4d16 | muse_moe_w4a16 | check | v2_nointerleave.py | 38e5d2371869cbc4 | passed | 32.69 ms |
| Oct 28 15:06 | 46001e6ab76847f5 | muse_bf16_proj | check | v3_hybrid.py | 400533fd066972f1 | passed | 4,411.0 µs |
| Oct 28 15:07 | 735b0e423e784074 | muse_bf16_proj | SUBMIT | v3_hybrid.py | 400533fd066972f1 | passed | 4,400.9 µs |
| Oct 28 15:09 | 9b15fa41af4b4fa5 | muse_attention_decode | check | v1_splitkv.py | a4bc14e395eb48f2 | failed |  |
| Oct 28 15:10 | 9f0b5c46219c49d3 | muse_moe_w4a16 | check | v3_sweep.py | b8e514c916581fc8 | passed | 46.77 ms |
| Oct 28 15:11 | 7ed6f3e7e75e4e43 | muse_attention_decode | check | v1b_splitkv.py | c0056616828add18 | passed | 4,033.4 µs |
| Oct 28 15:13 | fd8def23768d476c | muse_moe_w4a16 | check | v4_flatgrid.py | 8ec27a9cd71b988b | passed | 37.44 ms |
| Oct 28 15:18 | 62d4f8c2b2c248a0 | muse_attention_decode | check | v2_rows_hwfp8.py | 9188019cbf3ad831 | failed |  |
| Oct 28 15:18 | 9ea135a1581b4ea4 | muse_moe_w4a16 | check | v5_marlin_bsm_sweep.py | 63576ee056675309 | passed | 12.98 ms |
| Oct 28 15:20 | c0932f1836d747fc | muse_attention_decode | check | v2b_rows_hwfp8.py | cacc713284d02f5b | passed | 4,609.3 µs |
| Oct 28 15:20 | cf01732f11704d0b | muse_moe_w4a16 | check | v6_marlin_bsm_sweep2.py | 41fa2c5e394cc6d2 | passed | 12.67 ms |
| Oct 28 15:21 | ded1516bba6349bb | muse_decode_layer | check | v1_swaps.py | f42dbc7e497b87c1 | passed | 21.75 ms |
| Oct 28 15:27 | 0b6ae863caa241c9 | muse_moe_w4a16 | check | v8_marlin_bsm_sweep3.py | 152f2b8f2d92cf18 | passed | 12.60 ms |
| Oct 28 15:27 | 5e99533afa19424c | muse_attention_decode | check | v3_hybrid_explore.py | abae709fbf0f1d1e | passed | 3,730.1 µs |
| Oct 28 15:27 | 85cc211c7e5544d9 | muse_decode_layer | SUBMIT | v1_swaps.py | f42dbc7e497b87c1 | passed | 21.76 ms |
| Oct 28 15:27 | 933da6a5a7884130 | muse_moe_w4a16 | SUBMIT | v7_marlin_bsm_best.py | caf4deefff58c20b | passed | 12.19 ms |
| Oct 28 15:27 | e445d41c652140b7 | muse_decode_layer | check | v2_swaps_moebsm.py | 03a9096611fe35d2 | passed | 20.59 ms |
| Oct 28 15:33 | 0a602f52b5754048 | muse_attention_decode | check | v4_hybrid_explore.py | 665963767af8256a | passed | 3,588.6 µs |
| Oct 28 15:33 | 873a87c078fd4c27 | muse_decode_layer | SUBMIT | v2_swaps_moebsm.py | 03a9096611fe35d2 | passed | 20.61 ms |
| Oct 28 15:34 | 0712e3176b9f4b68 | muse_attention_decode | SUBMIT | v3b_best.py | 6c6ca1b42332a8ea | passed | 3,509.5 µs |
| Oct 28 15:43 | 2a5f698817174f51 | muse_bf16_proj | check | v4_router_largem.py | f72e923ad3c49f89 | passed | 4,685.0 µs |
| Oct 28 15:43 | 3c059475a33047a8 | muse_decode_layer | check | v3_attn_full.py | de2a59ba76015f19 | passed | 20.58 ms |
| Oct 28 15:43 | a08410ebc3b04bf7 | muse_attention_decode | SUBMIT | v5_best.py | cdf8cb0824680610 | passed | 3,504.5 µs |
| Oct 28 15:48 | 366b397f75a84a8f | muse_decode_layer | SUBMIT | v3_attn_full.py | de2a59ba76015f19 | passed | 20.41 ms |
| Oct 28 15:52 | af501bb211bb4695 | muse_moe_w4a16 | check | v9_postnorm_fused.py | 76bb16baff1fe0a6 | passed | 10.58 ms |
| Oct 28 15:53 | 04269b7477f74417 | muse_decode_layer | check | v4_postnorm_fused.py | 862d1ee4e2b58e94 | passed | 19.16 ms |
| Oct 28 15:54 | 1b5c613200e84954 | muse_moe_w4a16 | SUBMIT | v9_postnorm_fused.py | 76bb16baff1fe0a6 | passed | 10.59 ms |
| Oct 28 15:55 | 174d8047c762425e | muse_decode_layer | check | v5_router_fp32.py | 6ba58bceec67feb9 | passed | 19.61 ms |
| Oct 28 15:55 | 9cb6d6a409ff4cd6 | muse_decode_layer | SUBMIT | v4_postnorm_fused.py | 862d1ee4e2b58e94 | passed | 19.17 ms |
| Oct 28 15:57 | 172c76d646474db3 | muse_attention_decode | check | v6_sweepA.py | b462594b4779df8e | passed | 3,948.0 µs |
| Oct 28 16:03 | fd78956d0ecb409f | muse_attention_decode | check | v7_sweepB.py | 958a3e81852bc4c6 | passed | 4,007.0 µs |
| Oct 28 16:09 | bb656112f11e4320 | muse_attention_decode | check | v8_rows_occ.py | 193256efe4e05761 | passed | 3,355.1 µs |
