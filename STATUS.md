# STATUS — updated Oct 29 2026, ~2:50 AM ET

**Rules acceptance: 2026-10-28 ~6 PM ET, rules re-accepted by Arman personally via `okc login` (on his own
Mac); the box's earlier agent-accepted session was logged out and revoked. Submits from here use his token.**

Board (from `okc leaderboard`, Oct 29 ~2:27 AM ET; all provisional until the organisers' post-contest reruns):
- **Overall kernel board: rank 800** (top-1000 reached at ~7:45 PM ET). Prefill (`muse_prefill_layer`) has its own board.

| problem | best run (board row) | rank | job | file (sha256[:16]) |
|---|---|---|---|---|
| muse_bf16_proj | **1.115×** | 1136 | f0bd99103bf74d83 | v6_router_stacked.py (53c5c18702e1cb65) |
| muse_attention_decode | **1.317×** | 784 | efe18d6e4b8c4fe3 | v14b_docfix.py (f9b4b6409ce588ea) |
| muse_moe_w4a16 | **1.380×** | 705 | d625bb2af63b43c1 | v24_tuned.py (9cba0e2b…) |
| muse_decode_layer | **~1.24×** (board 1.246×; decode-32 full is noise on an unchanged path, r5 m1) | 715 | b35042c212354b1f | v12_precompile.py (b934cc00c842f1c2) |
| muse_prefill_layer | — | — | — | not started |

Candidate in review: decode_layer v13_fused_attn.py (check 8ddcc64985b5474e, **1.278× (1.259-1.298)**), PR #5.

MoE board reference (Oct 28 late): rank 50 / 300 / 500 / 700 / 800 at 2.37 / 1.97 / 1.74 / 1.38 / 1.31×.

PRs: #1 (audit-r2-fixes) merged at e4aacf1 (r3). #2 moe-cuda-mma (r4 APPROVED), #3 decode-moe-cuda (r5 APPROVED),
#4 attn-fused (r6 APPROVED): merge order #2 -> #3 -> #4, each retargeted to main, waiting on Arman. #5
decode-v13-fused (base attn-fused): v13, for review.

## Oct 31 4:26-5:37 PM ET: attention v22 series, 13 jobs (records for PR #20 review r1 M1; added Nov 2)

These were sent with `okc check` directly, not through tools/okc_send.py, so they had no sends.tsv rows. Their 13
rows are now in sends.tsv, in time order. Job id, file and send time come from okc's local job log; the server's
`created_at` matches it to the second. Outcomes come from `okc status <job> --json` (Nov 2, ~10:20 AM ET).
The .json/.txt profiles were restored and are committed. **The .log profiles were lost in the Oct 31 ~5:43 PM ET box
move (0 bytes)** and are not committed.

Counting: 12 checks of runs/v22_cuda_fv.py plus 1 diagnostic of runs/v22d_dbg.py (3934b18d) make 13 jobs and 9
profiles. The review's "12 jobs, 8 with profiles" counts v22_cuda_fv.py only.
- 8 scored, and b9586ff0 was the best of them.
- 3 failed correctness. Each was 14/16: tests 5 and 7, the two full-verify b >= 2 tests, failed, so nothing was timed.
- 1 was a judge duplicate and did not run.

The PR's earlier "one failed, two diagnostics" was wrong; there was 1 diagnostic and 3 failures. 51a55997 (76c1adc) was the
V^T head-offset bug fixed in 5039724.

sha256: okc's `content_hash` is not the file's sha256: for b9586ff0 it is 0021f802..., while the committed file is
8ee72590..., and the same holds for okc_send-recorded jobs. So the server cannot confirm file hashes. The sha256 values in
these rows are **inferred** from the file at the commit made in the same second as each send, and marked so in
sends.tsv. Supporting evidence: every committed job has a distinct content_hash, and the duplicate shares its original's.

Record caveats:
- **6d90f6e3e9784df6 (5:09:44 PM ET) was sent with no commit at its send time**: none falls between 6f46165
  (5:05:29 PM) and 10575fb (5:13:18 PM). The judge marked it `duplicate_of` 469a11da (same content_hash), so the bytes
  it sent were the same as 469a11da's (6f46165), and it consumed no run.
- **b9586ff0's commit time equals its send time** (5039724, 5:36:32 PM ET). The same is true of every committed v22
  job: each was committed in the same second as its send, or 1 s before.
- v22d_dbg.py (3934b18d) was never committed and the file is not on the box. Its sha256 is UNKNOWN; only okc's content_hash
  f9d0b032... is known.

| time (ET) | job | file | commit | okc content_hash[:12] | outcome |
|---|---|---|---|---|---|
| 16:26:20 | b0d7d09101e64fd2 | v22_cuda_fv.py | 179fdd4 | f0633f4366af | 1.254x (1.235-1.273), 16/16 |
| 16:34:54 | 6939558c645240ab | v22_cuda_fv.py | 93a54cb | 00c9410f8a87 | 1.267x (1.248-1.291), 16/16 |
| 16:44:02 | 833763a0bb304f7b | v22_cuda_fv.py | dceb08d | 6b4dd6856dd8 | FAILED correctness 14/16, not timed: test 5 (b2 ctx 100000 full verify page 1): rel L2 1.408e+00 > 0.15; test 7 (b4 ctx 6000 full verify edges): rel L2 1.416e+00 > 0.15 |
| 16:46:38 | 3934b18d3d704157 | v22d_dbg.py | uncommitted | f9d0b032dbdd | 1.337x (1.317-1.357), 16/16 |
| 16:52:56 | c8500ee5eeb04de7 | v22_cuda_fv.py | 3f0c0d5 | c89cc1cc2921 | FAILED correctness 14/16, not timed: test 5 (b2 ctx 100000 full verify page 1): rel L2 1.410e+00 > 0.15; test 7 (b4 ctx 6000 full verify edges): rel L2 1.408e+00 > 0.15 |
| 16:58:47 | defbcb21ce084f78 | v22_cuda_fv.py | d300793 | 20fc1d3cb8bd | 1.368x (1.347-1.388), 16/16 |
| 17:05:29 | 469a11daeb4749bb | v22_cuda_fv.py | 6f46165 | 55a165fc9506 | 1.381x (1.360-1.402), 16/16 |
| 17:09:44 | 6d90f6e3e9784df6 | v22_cuda_fv.py | uncommitted | 55a165fc9506 | not run: judge dedup, identical content to 469a11daeb4749bb |
| 17:13:19 | 87d219d56a4d4a6d | v22_cuda_fv.py | 10575fb | fa2db01fa1ab | 1.423x (1.401-1.444), 16/16 |
| 17:19:52 | 7a98fe19e6294071 | v22_cuda_fv.py | 7c77e6b | bc3509774be2 | 1.368x (1.348-1.389), 16/16 |
| 17:27:07 | 23257bf83b0e4dbd | v22_cuda_fv.py | b5a17a6 | a89f08368885 | 1.415x (1.394-1.436), 16/16 |
| 17:34:05 | 51a559973ef14097 | v22_cuda_fv.py | 76c1adc | c1623e126491 | FAILED correctness 14/16, not timed: test 5 (b2 ctx 100000 full verify page 1): rel L2 1.306e+00 > 0.15; test 7 (b4 ctx 6000 full verify edges): NaN/Inf |
| 17:36:32 | b9586ff0a9db47f0 | v22_cuda_fv.py | 5039724 | 0021f80226b7 | 1.433x (1.412-1.460), 16/16 |

## Oct 31 ~6:00-7:00 PM ET: attention v22 (PR #20), review r1 fixes (CHANGES REQUESTED by Senior Larper)

What v22 really gained (review m1). Check b9586ff0a9db47f0 scored **1.433x (95% CI 1.412-1.460)** against v20b's
1.406x, but:
- **The kernel alone is about +0.7%** (review r1's estimate): the two full-verify b >= 2 shapes the CUDA kernel
  replaces move the overall 1.4057 -> 1.4154 when the other shapes are held at v20b's times.
- **Most of 1.433 is noise**: shapes v22 does not touch (dp heads, b1, sliding, decode; byte-identical v20b code)
  ran 0.7-3.9% faster in that run than in v20b's check.
- **Selection**: b9586ff0 was the best of 8 scored v22 runs (12 v22_cuda_fv.py checks in all: 8 scored, 3 failed
  correctness, 1 judge duplicate; see the records section above). Its CI lower bound clears the board's CI upper bound 1.409 by only ~0.0028.
- So a re-check of v22 may well land near ~1.41-1.42x (an estimate from the kernel-only figure), i.e. not clearly
  above 1.409.

Review fixes (branch attn-v22, after merging origin/main a1860cc):
- m3: NVRTC compile + CuPy module load moved out of the first kernel_fn call. `_fv_cubin()` (host-only NVRTC) and
  `_fv_fn(dev)` (per-device load) run at import when an sm_90 GPU is present; `_fv_fn` raises instead of compiling
  if it is ever reached while a stream is capturing. `_fv_ok` also requires an sm_90 device (the cubin is sm_90a only).
- Nit: launches run under `cp.cuda.Device(q's device)` on that device's current torch stream (device guard).
- Nits: tools/attn_v22_fv.cu header rewritten to the shipped design (SS QK wgmma with Q in shared memory, n72 P.V with
  ones rows for the row sums, overlap); dead `WG_QK` macro and unused `wg_commit_wait` removed. The cubin is unchanged
  (byte-identical under NVRTC 13.4).
- Register figure is NVRTC-version dependent (nit): `tools/tests/attn_v22_nvrtc_cc.py`, sm_90a, `-std=c++17`:
  **NVRTC 12.9 (nvidia-cuda-nvrtc-cu12 12.9.86): 106 regs, 0 spill, 0 stack; NVRTC 13.4 (nvidia-cuda-nvrtc 13.4.92):
  104 regs, 0 spill, 0 stack.** The PR's "106 regs" matches 12.9. The judge's NVRTC version is not recorded.
- m2: `tools/tests/attn_v22_fv_kvscale.py`: full verify q_len 8, tp8, b 2-4, kv_scale 0.015625 / 0.37 / 2.5 and
  v_scale overrides 0.05 / 0.6 (K and V scales differ). Part A (CPU) emulates the kernel's tile loop step by step
  (mask, exp2 online softmax with the +8 bias, P rounded to E4M3, ones-row sums, v_scale / l, lse - 8) and feeds the
  submission's own Triton combine: 4/4 pass (rel. L2 0.069-0.079 vs bounds 0.150; the reference's own FP8-query
  error is 0.069-0.079). Negative control: dropping k_scale fails all 4 (0.31-0.52). A uniform v_scale error cancels
  in the per-head RMSNorm, so v_scale is only checked structurally. Part B (`ATTN_GPU=1`, sm_90 only) runs the real
  CUDA path, asserts it was taken, and replays it in a CUDA graph; it has NOT run yet (no H100/H200 off the judge).
- Nit: `/tmp/attn_cmp19.py` replaced by `tools/tests/attn_cpu_cmp.py`: v20b vs v22 CPU outputs and new cache rows
  byte-identical on 9 cases (maxdiff 0).
- CPU interpreter localtest (`tools/localtest_attn.py` on v22): every case fails only the new-cache 8% check, identically
  for the old (8ee72590) and new builds: the Triton 3.9 interpreter's fp32 -> E4M3 cast bug (r6 n2 caveat above), not v22.
- Offline Triton sm_90 harness (`SM90_GPU_TYPES=1 tools/tests/triton_sm90_compile.py runs/v22_cuda_fv.py`, run from
  muse_attention_decode/): 0 compile failures.

Submit file: muse_attention_decode/runs/v22_cuda_fv.py rebuilt by tools/build_attn_v22.py, **sha256 d2925cb84321faeac15c47620687cb9059559445fa92c72dffcfe616d5b89d60**
(was 8ee72590baa893fc..., the file checked as b9586ff0). The kernel code is unchanged but the bytes changed, so a
new check is needed before any submit (not sent: no OKC_TOKEN on the box after the Oct 31 box move).

## Oct 29 ~3:40-3:45 PM ET: prefill v2b and decode v19 submitted (review round 13 approvals of #16 / #17)
- prefill_layer v2b_dequant_grouped.py (sha d0b2cc11..., sent at 99e0c27): submit 7b69b2fcfffb40dd, **1.088x (1.072-1.105)**,
  5/5 tests. First prefill board entry. Per TASK.md, prefill has its own board and does not count toward the overall rank.
  First-turn shapes 1.186-1.310x; turn 1 / turn 4 shapes ~1.003-1.004x (Marlin path, unchanged).
- decode_layer v19_nb2_256_evict.py (sha d70c6c3e..., sent at 1e4fd5c): submit 7702cb83b35b4737, **1.364x (1.344-1.385)**,
  13/13 tests. The CI lower bound clears v17's board upper bound (1.337), so this is a significant improvement over v17
  (1.318x). verify 16 sliding / full 1.343 / 1.237x, verify 32 1.282 / 1.182x.
- Public board after both (~3:44 PM ET): overall rank 699 of 1606 (solved 4; prefill not counted).
- Ledger note (review round 13, #15): 17 older rows on main fail the ledger check. 16 of them have commit time equal to
  send time, i.e. they were back-filled after the fact. The v7_audit_nits row has commit 'dedupe' and a sha256 that does
  not match the file. They are left as recorded and not rewritten; treat them as unverified history.

## Why profile .log files are empty (review round 13)
The `profiles/<file>.<job>.log` that okc saves next to each run holds the last 16,000 characters the submission
printed (`log_tail` in `okc status --json`). It is not a judge log. Our kernels print nothing: the BENCH
exploration code that printed timings was removed in review round 10, and printing inside kernel_fn is not allowed
(a tensor print would sync). So the .log files are empty (0 bytes), and an empty one means "no output". It does
not mean a capture failure. tools/okc_send.py cannot add anything, because the file holds the run's own stdout.
Options, not done: a one-line prepare()-time print in future kernel versions (torch / triton versions, path flags
such as FASTSCALE or grouped_ok), which would make the .log useful for checking the judge's environment, e.g.
torch._grouped_mm availability. That changes the kernel file, so it belongs in a new vN, not a retro-edit.

## Oct 29 ~3:46-4:01 PM ET: decode v20 / attention v17 (branch decode-v20; folded into main via status-r13)
- tools/rows_prefetch_patch.py (shared), tools/build_decode_v20.py, tools/build_attn_v17.py. The full-decode
  (batch >= 2) split-KV "rows" kernel runs on 128-key tiles and loads the next tile's page indices one iteration
  ahead (PREF). Plus #17 nits: the NB 2 comment now says "from 81 tokens"; the dead NB == 4 MINB branch is removed.
- decode v20 check 5fa280a493f244a3: 1.366x (1.345-1.386); rows kernel 257 -> 245 us at decode 32 full, b16 flat.
  attention v17 check 8deec285efbe4b57: 1.398x (1.377-1.424); decode 32 full 264 -> 249 us. Neither clears its PR
  bar (decode 1.385, attention 1.409). The pages are randomly permuted, so each key is two random 128-byte reads;
  ~2.6 TB/s is near what that access pattern gives.

## Oct 29 ~2:36-3:00 PM ET: leaderboard gaps; decode v18 / v18b; MoE v26e (Arman: climb incrementally, keep afloat)
Leaderboard (public board, ~2:40 PM ET; overall rank 754 / 1589):

| problem | ours | rank | top-1 | top-50 | ~rank 500 | rank at +0.05x |
|---|---|---|---|---|---|---|
| bf16_proj | 1.115 | 1224 / 1378 | 1.637 | 1.560 | 1.286 | 1025 |
| moe_w4a16 | 1.500 | 694 / 1105 | 2.689 | 2.386 | 1.793 | 677 |
| attention_decode | 1.385 | 793 / 1058 | 3.054 | 2.639 | 1.733 | 746 |
| decode_layer | 1.318 | 694 / 972 | 2.418 | 2.184 | 1.541 | 645 |

The overall formula is unpublished. The best fit is the sum of per-problem step times (SGLang 4831 / 13448 / 3766 / 22620 us
for bf16 / MoE / attention / decode). Overall places per +0.05x on that model: decode ~23, MoE ~9, bf16 ~5,
attention ~3. So decode_layer (half the summed time) and MoE pay most per unit of effort.

decode v18 (branch decode-v18 off origin/decode-v17 032fef4, tools/build_decode_v18.py, 9bd11b9): CUDA experts for
128 / 256-token verify (NB 2 / NB 4), plus the review round 10 minors (FASTSCALE per state, eager compile, no BENCH).
Check 9566de62f7e94ed1 scored 1.333x (1.313-1.353): verify16 422 / 619 -> 405 / 587 us (win), but verify32 lost to
Marlin with NB 4 (115 registers). v18b (8e01dcf) caps the CUDA path at 192 tokens. Check ae89a962123748e6 scored
1.328x (1.308-1.348), in the noise of v17 (board 1.318, check 1.333): verify16 402 / 594 us. No PR.

MoE v26e (branch moe-v26e off origin/moe-v26c a89056a, tools/build_moe_v26e.py -> runs/v26e_nb2_evict.py, 252697a):
v26c + the same NB 2 CUDA experts at 128 tokens (SMALL_MAX 192), + L2 evict-first / L1 no-allocate weight and scale
loads (createpolicy + ld.global.nc...L2::cache_hint; kernel-wide policy register, NVRTC registers and spills equal
to v26c), + review round 10 minors. Check b8a324016b7b4422 scored **1.546x (1.523-1.569)**. Board v26c: 1.500x
(1.478-1.523).

| case | v26c check f479a34c | v26c submit 7eaa9622 | v26e b8a32401 |
|---|---|---|---|
| decode 1 | 40.0 | 38.5 | 39.2 |
| decode 16 | 120.4 | 120.0 | 117.7 |
| decode 32 | 165.2 | 158.6 | 161.8 |
| verify 1 (8 tok) | 93.3 | 87.9 | 86.4 |
| verify 16 (128 tok) | 338.6 | 349.9 | **298.1** |
| verify 32 (256 tok) | 468.7 | 468.3 | 470.8 |

(median us). The whole gain is at verify16 (NB 2, -12 to -15%). Evict-first is neutral on the bandwidth-bound
small shapes. Not submitted, no PR opened (for Arman / Larper).

Follow-ups (~3:05 PM ET, status-r11):
- Larper approved PR #14 (moe-v26e). Arman submitted v26e at 2:59 PM ET as job 94cff1d839e947df: 1.539x (1.516-1.562),
  5/5 tests. The CI lower bound does not clear the old board entry's upper bound (1.523), so this is not recorded as
  a significant improvement. It is the new board entry (point estimate +0.039x over v26c 1.500x).
- NB 2 GEMM-1 (gated, W8, MINB 3) under NVRTC -Xptxas -v: 80 registers, a 24-byte spill with FASTSCALE (16 bytes
  without), 33 KB shared. The NB 2 GEMM-2 kernel: 62 registers, no spill. Same in decode v18 / v18b and MoE v26e.
- sends.tsv rows and profiles for checks 9566de62f7e94ed1 / ae89a962123748e6 (decode v18 / v18b), b8a324016b7b4422
  (v26e), the v26e submit, and v26c f479a34c48984e24 / 7eaa9622f391455b are recorded here (review round 11 m1).
- moe-v26d is a stale offline experiment: cp.async activation staging, projected ~0%, never checked. It is kept for
  reference only; nothing builds on it.

## Quotas (UTC day Oct 29, i.e. since 8 PM ET Oct 28), from `okc problems` at ~2:53 PM ET (supersedes the ~2:45 PM line)
(Current as of ~2:53 PM ET. Later per-round counts are in each round's section; quotas reset at 8 PM ET.)
- Checks left: bf16 0 of 20 (resets 8 PM ET), attention 12, MoE 2 (after v26e b8a324016b7b4422), decode_layer 7
  (after v18 9566de62f7e94ed1 and v18b ae89a962123748e6), prefill_layer 20 (new 5th problem, not started).
- Scored runs left: bf16 10, attention 8, MoE 7 (after the v26e submit), decode_layer 7, prefill_layer 10. Today's submits (Arman): MoE v26e 94cff1d839e947df 1.539x, MoE v26c
  7eaa9622f391455b 1.500x, decode v17 6a55705a372f40ac 1.318x (and earlier attention v15b 8f7ac0501b5849d7 1.385x).

## HISTORICAL (superseded by the Quotas section above; kept unedited per the no-silent-deletion rule): Quotas (UTC day Oct 29, i.e. since 8 PM ET Oct 28), from sends.tsv (~3:27 AM ET)
- Checks used: bf16 19 of 20 (v27 x2 this session; 2 pre-check rejections created no job); attention 3 of 20; MoE 15 of 20 (v25, v25b this session); decode_layer 6 of 20
- Checks used: bf16 17 of 20; attention 5 of 20 (v15a, v15b this session); MoE 15 of 20 (v25, v25b); decode_layer 6 of 20
  (v13 re-send 9a5880 and bf16 v7 re-send 9e6108 deduped by the judge).
- Submits used: MoE 1 (v24, d625bb2af63b43c1, 1.380×), decode_layer 2 (v12 b35042c212354b1f 1.246×; v13
  093c2ebc301344e8 1.251×), attention 1 (v14b, efe18d6e4b8c4fe3, 1.317×), bf16 0.
- Note: the judge dedups identical code, so a byte-identical re-check can't be used to replicate a timing.

(Ported from branches moe-v26c / moe-v26d by status-r11, review round 11.)

## Oct 29 ~4:57-5:05 AM ET: MoE v26d, activation staging (branch moe-v26d off origin/moe-v26c 377f1f0). Offline only, no check.
- Quantified before building (bs32, GEMM 1 CTA = 8 warps, KS 1024, 16 k-tiles per warp; GEMM 2 alike):
  - L2 bytes: ~2560 CTAs x 16 KB per GEMM, ~82 MB per call over ~160 us = ~0.5 TB/s, ~5% of L2 bandwidth: not a cost.
  - Issue: staging is ~4 LDG + 4 STS + index math (~40 instr) per thread vs ~3700 in the k-loop (16 tiles x ~115 x
    gate+up): ~1%.
  - Latency: the lst -> x loads (two dependent L2 hits, ~1 us) already overlap the first PF weight loads (HBM, same order),
    and 4 CTAs/SM hide the rest. Register-resident rows would need ~128 B-fragment regs per lane; sharing across more
    N-tiles per CTA (wider W) does not cut per-thread issue.
  - Expected: ~0-2% of moe_mma, ~0.5-1.5% overall at bs8-32 (less at bs1); clearing 1.52 needs ~+3%. Not promising.
- Built the cheapest exact variant anyway: cp.async.cg 16-byte staging straight to smem (src-size 0 zero-fill), one
  commit group waited before the barrier; same bytes, bit-identical to v26c. tools/build_moe_v26d.py ->
  muse_moe_w4a16/runs/v26d_async_stage.py, byte-exact at HEAD. NVRTC: -2 PTX instructions per kernel (3260 -> 3258,
  1732 -> 1730), same registers (64/54 at 8 warps, 72/52 at 4) and spills. Confirms ~0 gain; check not spent.
- Where MoE small-batch time really is now: moe_mma's k-loop issue (dequant ~9 ALU + 4 HMUL2 + 2 splat per 32-bit code
  word, near the bit-layout minimum per the v26b analysis), post-norm 3-4 us, route_prep 2-3 us. Bigger next levers would be
  structural: fp16 instead of bf16 MMA operands (HMUL2 f16 + e2m1 -> f16 bit layout with more slack; precision delta
  needs analysis), or a combined GEMM1+GEMM2 persistent kernel to remove one more launch + tail (~2-3 us).

## Oct 29 ~4:46-5:00 AM ET: MoE v26c (branch moe-v26c off origin/moe-v26 325eb72; built from v26a)
- Gap census from v26a's profiles (bs 1/8/16/32): moe_mma 36.0/69.8/93.2/157.5 us; outside it silu_mul 2.7-3.8, post-norm
  3.4-3.7, zero-fill 0.9-3.6, route_prep 1.4-1.6 us, plus 5 inter-kernel gaps: ~9-13 us per call (6-8% at bs16-32,
  ~20% at bs1). Activation staging sits inside moe_mma (not separable without a profiler) and needs a loop restructure, so
  the launch chain went first.
- v26c: 6 -> 4 launches. FP32 sum buffers torch.empty; route_prep runs one block per route and zeroes local routes' GEMM-1
  (2I) and GEMM-2 (H) rows (block 0 builds the lists). silu_mul removed: GEMM 1 split-K CTAs count arrivals per (work item,
  row block) in self-resetting int32 counters (state["c1"], zeroed in prepare, reset by the last CTA), and the last CTA
  writes silu(gate) * up as BF16 for its 16 * WARPS rows with silu_mul's exact expression. Same FP32 sums and SiLU math:
  bit-identical to v26a up to the FP32 atomic order both already have. __launch_bounds__ minBlocks keeps v26a's occupancy
  for GATED (8 warps: 4 CTAs/SM at 64 regs, 8 B spill vs v26a's 28 B; 4 warps: 7 CTAs/SM, 72 regs).
  tools/build_moe_v26c.py -> muse_moe_w4a16/runs/v26c_fused_chain.py, byte-exact at HEAD 8bd9599.
- Check f479a34c48984e24 (sha 85fe5e71..., commit 8bd9599): 5/5, 1.497x (95% 1.474-1.519) vs v26a 1.457x (1.435-1.479).
  | load | SGLang | v26c us / x | v26a x |
  |---|---|---|---|
  | decode 1 | 79.3 | 40.0 / 1.982 | 1.914 |
  | decode 16 | 188.0 | 120.4 / 1.561 | 1.502 |
  | decode 32 | 206.4 | 165.2 / 1.249 | 1.219 |
  | verify 1 (8 tok) | 151.1 | 93.3 / 1.619 | 1.542 |
  | verify 16 (128, Marlin) | 464.2 | 338.6 / 1.371 | 1.361 |
  | verify 32 (256, Marlin) | 614.5 | 468.7 / 1.311 | 1.302 |
  Lower bound 1.474 is not > 1.48 (bar: v26a's CI upper 1.479), but Arman opened PR #11 (base moe-v26) and submitted
  it: job 7eaa9622f391455b, 1.500x (CI 1.478-1.523), record a89056a on moe-v26c. Second MoE check kept.
- Left outside moe_mma: post-norm sum 3.0-4.1 us, route_prep 2.0-2.8 us. Folding the norm into GEMM 2 needs every local
  route of a token complete (rows finish in different items): a per-token finisher serializes up to 8 tokens' norms in the
  last CTA (tail could exceed the 3-4 us kernel); the per-route variant (route-level norm + FP32 atomics into out, per-token
  counter converts to BF16) nets ~1-2 us and changes the sum order. PDL can't prefetch weights before griddepcontrol.wait
  (the weight address depends on route_prep's work list), so it would only hide ~1 us of launch latency per boundary. Neither
  is clearly enough for a check; next real lever is activation staging inside moe_mma (16 KB per CTA from L2 before the loop).
- (decode v16's STATUS section lives on branch decode-v16 / PR #10.)

## Oct 29 ~4:52-5:05 AM ET: decode v17 (branch decode-v17 off origin/decode-v16 b8d806d)
- v17 = v16 + MoE v26c's launch-chain fusion in the layer's CUDA expert path, brought in by building from source: the
  moe_mma / route_prep CUDA sources of v16 and MoE v26a are identical, so tools/build_decode_v17.py applies v26c's
  CUDA-side replacements verbatim (resulting _CU_SRC == MoE v26c's, checked) plus the layer's host-side edits (empty FP32
  buffers, route_prep grid NR, c1 counters in state["moe_small"], silu_mul launch dropped). 6 -> 4 MoE launches,
  bit-identical math. Graph replay: each (work item, row block) counter is reset to 0 by its last CTA inside the same
  launch; c1 is zeroed in prepare, so every call / replay starts from zeros (one MoE per kernel_fn, launches serialized on
  the stream).
- 17-32 tokens (decode 32) now use the CUDA experts (standalone 32-token config NB 1, (8, 4, 2)): projected from v15b
  (CUDA@32 ~6-11 us behind Marlin) minus v26a (~-10 us at bs32) and v26c (~-10 us): ~-10 to -14 us vs Marlin.
- Byte-exact at HEAD 617b28c; NVRTC: GATED 64 regs / 8 B spill (8 warps), 72 regs (4 warps).
- Check 5dbee2e16e3741f9 (sha 67dbbe14..., commit 617b28c): 13/13, 1.333x (95% 1.313-1.353) vs v16 1.289x (1.270-1.309),
  v14 1.278x, board v13 1.251x. Lower bound > 1.31 -> PR #12 against decode-v16. Submitted later by Arman: job
  6a55705a372f40ac, 1.318x (CI 1.298-1.337), record 032fef4.
  | case | SGLang | v16 us / x | v17 us / x |
  |---|---|---|---|
  | decode 1 sliding | 172.9 | 117.0 / 1.462 | 113.9 / 1.518 |
  | decode 1 full | 180.9 | 125.0 / 1.471 | 121.4 / 1.490 |
  | decode 16 sliding | 256.5 | 190.7 / 1.396 | 180.7 / 1.419 |
  | decode 16 full | 434.0 | 317.6 / 1.362 | 312.8 / 1.387 |
  | decode 32 sliding | 257.3 | 236.5 / 1.096 | 218.2 / 1.179 |
  | decode 32 full | 619.8 | 486.4 / 1.262 | 468.4 / 1.323 |
  | verify 1 sliding | 262.5 | 176.0 / 1.483 | 172.9 / 1.518 |
  | verify 1 full | 280.9 | 187.8 / 1.476 | 185.5 / 1.515 |
  | verify 16 sliding | 513.2 | 434.9 / 1.155 | 422.4 / 1.215 |
  | verify 16 full | 718.2 | 619.5 / 1.110 | 619.0 / 1.160 |
  | verify 32 sliding | 722.4 | 610.6 / 1.180 | 599.9 / 1.204 |
  | verify 32 full | 1,083.3 | 960.1 / 1.097 | 971.0 / 1.116 |
  decode 32 gained 18 us per layer as projected (CUDA experts at 32 tokens); verify 16/32 (Marlin, unchanged code) moved
  with SGLang's own timing (noise).

## Oct 29 ~4:46-5:00 AM ET: MoE v26c (branch moe-v26c off origin/moe-v26 325eb72; built from v26a)
- Gap census from v26a's profiles (bs 1/8/16/32): moe_mma 36.0/69.8/93.2/157.5 us; outside it silu_mul 2.7-3.8, post-norm
  3.4-3.7, zero-fill 0.9-3.6, route_prep 1.4-1.6 us, plus 5 inter-kernel gaps: ~9-13 us per call (6-8% at bs16-32,
  ~20% at bs1). Activation staging sits inside moe_mma (not separable without a profiler) and needs a loop restructure, so
  the launch chain went first.
- v26c: 6 -> 4 launches. FP32 sum buffers torch.empty; route_prep runs one block per route and zeroes local routes' GEMM-1
  (2I) and GEMM-2 (H) rows (block 0 builds the lists). silu_mul removed: GEMM 1 split-K CTAs count arrivals per (work item,
  row block) in self-resetting int32 counters (state["c1"], zeroed in prepare, reset by the last CTA), and the last CTA
  writes silu(gate) * up as BF16 for its 16 * WARPS rows with silu_mul's exact expression. Same FP32 sums and SiLU math:
  bit-identical to v26a up to the FP32 atomic order both already have. __launch_bounds__ minBlocks keeps v26a's occupancy
  for GATED (8 warps: 4 CTAs/SM at 64 regs, 8 B spill vs v26a's 28 B; 4 warps: 7 CTAs/SM, 72 regs).
  tools/build_moe_v26c.py -> muse_moe_w4a16/runs/v26c_fused_chain.py, byte-exact at HEAD 8bd9599.
- Check f479a34c48984e24 (sha 85fe5e71..., commit 8bd9599): 5/5, 1.497x (95% 1.474-1.519) vs v26a 1.457x (1.435-1.479).
  | load | SGLang | v26c us / x | v26a x |
  |---|---|---|---|
  | decode 1 | 79.3 | 40.0 / 1.982 | 1.914 |
  | decode 16 | 188.0 | 120.4 / 1.561 | 1.502 |
  | decode 32 | 206.4 | 165.2 / 1.249 | 1.219 |
  | verify 1 (8 tok) | 151.1 | 93.3 / 1.619 | 1.542 |
  | verify 16 (128, Marlin) | 464.2 | 338.6 / 1.371 | 1.361 |
  | verify 32 (256, Marlin) | 614.5 | 468.7 / 1.311 | 1.302 |
  Lower bound 1.474 is not > 1.48 (bar: v26a's CI upper 1.479): no PR #11. Not submitted. Second MoE check kept.
- Left outside moe_mma: post-norm sum 3.0-4.1 us, route_prep 2.0-2.8 us. Folding the norm into GEMM 2 needs every local
  route of a token complete (rows finish in different items): a per-token finisher serializes up to 8 tokens' norms in the
  last CTA (tail could exceed the 3-4 us kernel); the per-route variant (route-level norm + FP32 atomics into out, per-token
  counter converts to BF16) nets ~1-2 us and changes the sum order. PDL can't prefetch weights before griddepcontrol.wait
  (the weight address depends on route_prep's work list), so it would only hide ~1 us of launch latency per boundary. Neither
  is clearly enough for a check; next real lever is activation staging inside moe_mma (16 KB per CTA from L2 before the loop).
- (decode v16's STATUS section lives on branch decode-v16 / PR #10.)
## Oct 29 ~4:41-4:50 AM ET: decode v16 (branch decode-v16 off origin/moe-v26 325eb72)
- decode-v14's files brought in by file checkout (git checkout origin/decode-v14 -- muse_decode_layer/runs/v14_verify_split.py
  tools/build_decode_v14.py), committed as-is; no merge of the decode-v14 branch.
- v16 = v14 + MoE v26a's exact PRMT + IMAD E4M3 scale splat in the layer's CUDA expert kernels (same 0x08..0x7E range check
  in _moe_small_bank, v14 cvt fallback). tools/build_decode_v16.py -> muse_decode_layer/runs/v16_int_splat.py, byte-exact at
  HEAD 7bdc9bc; NVRTC: GATED moe_mma PTX 4194 -> 3171, down 2243 -> 1732.
- Check b94ed32acfce4452 (sha a1057a43..., commit 7bdc9bc): 13/13, 1.289x (95% 1.270-1.309) vs v14 1.278x (1.259-1.297),
  board v13 1.251x. Lower bound 1.270 is not > 1.28: no PR #10. Not submitted.
  | case | v14 us / x | v16 us / x |
  |---|---|---|
  | decode 1 sliding | 119.4 / 1.438 | 117.0 / 1.462 |
  | decode 1 full | 126.1 / 1.451 | 125.0 / 1.471 |
  | decode 16 sliding | 192.9 / 1.340 | 190.7 / 1.396 |
  | decode 16 full | 325.4 / 1.327 | 317.6 / 1.362 |
  | decode 32 sliding | 232.6 / 1.098 | 236.5 / 1.096 |
  | decode 32 full | 480.0 / 1.278 | 486.4 / 1.262 |
  | verify 1 sliding | 187.7 / 1.415 | 176.0 / 1.483 |
  | verify 1 full | 199.0 / 1.409 | 187.8 / 1.476 |
  | verify 16 sliding | 443.6 / 1.210 | 434.9 / 1.155 |
  | verify 16 full | 624.1 / 1.145 | 619.5 / 1.110 |
  | verify 32 sliding | 618.8 / 1.167 | 610.6 / 1.180 |
  | verify 32 full | 973.5 / 1.111 | 960.1 / 1.097 |
  Gains where the CUDA experts run (<= 16 tokens: decode 1/16, verify 1 = 8 tokens, -2 to -12 us); 32+ tokens run Marlin
  (unchanged; differences are noise).
- Second decode check NOT spent: the only fresh lever is CUDA experts at 17-32 tokens; v15b measured that path ~6 us slower than
  Marlin at decode 32 (238.5 vs 232.6 us, old splat, W16), and the splat saved ~10 us at bs32 standalone, so projected ~228 vs
  ~233-237 us: ~2% on 2 of 12 shapes, ~0.3% overall, not enough to clear 1.28. Worth folding into a future decode bundle.
- MoE v26b (4-bit unpack), offline analysis only, no check: each 32-bit code word -> 4 bf16x2 A registers costs 9 ALU ops
  (3 x SHF.W + LOP3, 1 x 3 ops) + 4 HMUL2. The e2m1 bits must sit at bf16 bits {15, 8, 7, 6} (sign, 2 exponent LSBs, mantissa
  MSB: bf16/e2m1 = 2^-126 on both the subnormal and normal codes, the HMUL2 by scale * 2^119 fixes the bias), so a 1-op
  in-place output plus three 2-op outputs needs four rotations of {6,7,8,15} tiling Z16; a hand search found none. Best case is
  ~1 op per word (~3% of the k-step): not clearly promising, MoE check kept. Bigger bs32 levers left: the 5 launches' gaps
  (GEMM 1 -> silu -> GEMM 2 tails) and per-CTA activation staging (16 KB per CTA, ~22% of its weight bytes, from L2).

## Oct 29 ~4:35-5:00 AM ET: MoE v26a (branch moe-v26 off origin/bf16-wgmma 0b2cd87; moe-v25 is its ancestor)
- bf16 wgmma is PARKED (Arman) until a cluster/DSMEM-reduction or producer-warp design; v28/v29 notes below.
- Where small-batch MoE time goes (judge routing, generate_input dist "normal", busiest GPU, 200 CPU seeds): bs32 touches
  16.2 local experts (10-23), 50 routes, hottest expert 14.7 tokens, half the touched experts get 1 token; bs16 12.1 experts,
  bs8 8.5, bs1 2.6. At 28.3 MB of NVFP4 per expert, v24/v25b ran ~10.6-11.7 us per expert (~2.5-2.7 TB/s) at bs8-32, so
  the gap is weight-streaming efficiency at every small bs, not a bs32-only issue. v23's in-prepare bench (normal routing, not
  skewed) showed NB 2 costing +8% (bs32) / +11% (bs16): extra MMAs + B loads alone slow it, so moe_mma is issue-bound.
- Fix: the E4M3 scale splat (8 per 16 x 64 tile, ~5 ops each) becomes PRMT + IMAD: for a positive normal byte b,
  bf16(e4m3(b) * 2^119) == (b << 4) + (239 << 7) per half, exact (checked on CPU for all 119 bytes 0x08..0x7E, and the
  macro with byte_perm emulated on 20k random scale words). prepare() checks every local scale byte is 0x08..0x7E (judge
  weights: clamped to [8, 448], or >= 2^-6 with wide scales) else compiles v24's cvt path. Output bit-identical to v24's.
  GATED moe_mma PTX 4194 -> 3171 instructions (-24%), 0 spills, same registers. tools/build_moe_v26.py -> v26a, byte-exact.
- Check bbebbbdf181f4934 (sha 196c6708..., commit 05d6779): 5/5, 1.457x (95% 1.435-1.479) vs board v24 1.380x, v25b 1.391x.
  | load | SGLang | v26a | x | v25b x |
  |---|---|---|---|---|
  | decode 1 | 79.7 | 41.7 | 1.914 | 1.809 |
  | decode 16 | 194.0 | 129.2 | 1.502 | 1.391 |
  | decode 32 | 213.4 | 175.0 | 1.219 | 1.117 |
  | verify 1 (8 tok) | 150.4 | 97.6 | 1.542 | 1.423 |
  | verify 16 (128, Marlin) | 463.0 | 340.1 | 1.361 | 1.371 |
  | verify 32 (256, Marlin) | 617.3 | 474.1 | 1.302 | 1.322 |
  CI lower 1.435 > 1.40 -> PR #9 opened against bf16-wgmma (the branch base); not submitted. Remaining headroom: bs32 still
  ~10.9 us per expert; next levers are the E2M1 code unpack (rotl + mask per a-register) and launch gaps (5 launches).

## Oct 29 ~4:30-5:00 AM ET: bf16 v28 checked (Arman: spend the expiring check), v29 built offline (not sent)
- v28 check c20bde96ace6489d (sha 3613a8de...; file committed in 56d3628, sent at HEAD cd5a674 per sends.tsv): 9/9 passed, 0.915x (95% 0.901-0.928); v27 was 0.927x, v17 1.133x.
  m=8-32 per shape, median us / speedup (v17 | v27 | v28):
  | case | m | v17 | v27 | v28 |
  |---|---|---|---|---|
  | verify 1 qkv_gate | 8 | 14.0/0.991 | 14.6/0.954 | 17.0/0.838 |
  | verify 1 o | 8 | 6.4/1.079 | 9.3/0.702 | 9.0/0.734 |
  | verify 1 pooler | 8 | 20.3/1.071 | 21.7/0.973 | 24.5/0.864 |
  | verify 1 unpooler (n 1024) | 8 | 6.8/1.020 | 7.0/0.972 | 6.6/1.018 |
  | decode 16 qkv_gate | 16 | 13.9/1.005 | 21.5/0.659 | 21.2/0.655 |
  | decode 16 o | 16 | 6.3/1.084 | 11.9/0.555 | 11.7/0.572 |
  | decode 16 pooler | 16 | 20.0/1.069 | 32.2/0.664 | 30.0/0.714 |
  | decode 16 unpooler | 16 | 18.2/1.029 | 32.3/0.582 | 29.7/0.611 |
  | decode 32 qkv_gate | 32 | 14.0/1.007 | 26.8/0.540 | 30.0/0.479 |
  | decode 32 o | 32 | 6.3/1.070 | 16.5/0.427 | 16.5/0.414 |
  | decode 32 pooler | 32 | 20.4/1.069 | 38.1/0.570 | 42.4/0.509 |
  | decode 32 unpooler | 32 | 18.1/1.033 | 37.5/0.487 | 41.5/0.449 |
  Other shapes run v17 routes and match v17. No m=8-32 shape beats v17.
- Reading: the pipelining (all loads in flight) bought nothing visible, and where v28 doubled S (qkv/pooler 16->32 and 8->32)
  it got slower, growing with m. Time vs m fits a split-K atomic cost of S*M*N FP32 atomics at roughly 110-180 G/s
  (qkv m 8->32: +13 us for +1.8 M atomics; pooler +18 us for +3.1 M; o +7.5 us for +0.8 M). Extrapolated atomic-free base:
  qkv ~12.7 (v17 14.0), pooler ~18.5 (20.3), o ~6.5 (6.4), unpooler ~17.9 (18.2) us.
- v29 (tools/bf16_v29_wgmma.cu, tools/build_bf16_v29.py -> muse_bf16_proj/runs/v29_wgmma_partials.py, built from v28,
  byte-exact at HEAD 3a95c60, sha 7ca4ece7...): atomics replaced by plain __stcg partial stores ws[s][tok][n] ([S][32][N]
  FP32) + last-CTA (counter) reduction summing s = 0..S-1 in a fixed order (float4 over 4 rows, 4 accumulators),
  deterministic, no ws zeroing; packed 8-byte BF16 stores. Verified offline: NVRTC sm_90a on the embedded source, 0 spills
  (56 regs at NT 8/16/32); CPU emulation tools/tests/bf16_v29_partials_emu.py exact over M 5-32 and 3 geometries with
  random CTA completion order, stale-garbage ws (never read), counter back to 0, and back-to-back calls reusing ws; CPU
  tests 13/13 (the CPU path never runs the kernel).
- v29 routing: per (layer, m) via WG_ROUTE, which is EMPTY. No measured v28 win, and no clear projected win: v29's
  last-CTA reduction reads S*64*M partials serially (qkv S 32: 64 KB per tile at m 8, 256 KB at m 32), projected
  ~1.5 us / ~4 us tail plus partial traffic. Projections: verify-1 pooler ~20.5 vs v17 20.3, qkv ~14.7 vs 14.0 (ties at best),
  m 16/32 worse. With the route empty, prepare() skips the wgmma repack/workspaces/compile and v29 behaves as v17 (do not check it).
- Next if bf16 is revisited: the reduction must stay distributed and cheap (cluster/DSMEM reduction of the S splits, or
  fewer splits with deeper per-CTA pipelines, i.e. producer warp + full/empty barriers), and per-CTA overhead needs a
  micro-profile. Measured wgmma base time only beats v17 by ~9% on qkv/pooler even with a free reduction. Fresh bf16 quota
  after 8 PM ET; v29 with WG_ROUTE = {("pooler", 8), ("qkv_gate", 8)} is the cheapest single measurement of the new reduction.

## Oct 29 ~4:25-4:55 AM ET: bf16 v28 pipelined wgmma (branch bf16-wgmma). SUPERSEDED: Arman then had the check spent
(c20bde96ace6489d, 0.915x; see the ~4:30-5:00 AM section above). Kept for the design notes; its "check saved" plan is moot.
v28_wgmma_pipe.py, built by tools/build_bf16_v28.py from v17 + tools/bf16_v28_wgmma.cu (byte-exact at HEAD).
- Change vs v27: each CTA's k range is NCH = 4 64-k chunks with one W stage per chunk (no slot reuse), all 4 TMA loads issued up
  front, wgmma over the chunks with a single wait_group(0) at the end (v27 drained after every chunk). Split-K raised so NCH = 4 on
  every layer (qkv/pooler S 32, unpooler S 16, o S 4): 512-2048 CTAs, 36-48 KB smem -> 4-6 CTAs/SM, ~128-190 KB in flight per SM.
- Verified offline: NVRTC sm_90a + ptxas, no spills (32/34/42 regs at NT 8/16/32); CPU addressing emulation exact (M 5-32, NCH 4,
  two geometries); CPU test path 13/13. No slot is reused, so there is no stage-overwrite hazard to get wrong.
- Why I didn't use a single warpgroup with stage reuse + deferred waits: with one warpgroup, a W stage can't be refilled until the
  wgmma group reading it has retired, and that group is the newest one, so the refill still needs wait_group(0) (v27's stall).
  Decoupling them needs the producer warp + full/empty mbarriers (warp specialization), whose barrier protocol the addressing
  emulation can't validate. v28 avoids reuse instead.
- Why the check was saved: v27 lost more as m grew (m=8 0.70-0.97x, m=32 0.43-0.57x). The likely m-dependent cost is the split-K
  reduction: S x M x N FP32 atomics (qkv m=32: 1.2 M at S=16 in v27, 2.4 M at S=32 in v28), which v28 makes larger. Chance of a
  CI lower bound > 1.14 judged low, so per the rule the last bf16 check of the day is kept for after 8 PM ET.
- Plan for the fresh quota (8 PM ET): (1) send v28 as is: its per-shape table separates the pipeline gain (m=8) from the
  reduction cost (m=32); (2) then replace the atomics with deterministic plain partial stores ws[s][tok][n] + a last-CTA sum, or
  cut S with all-resident stages (S 16 -> NCH 8, 64 KB smem), routing per shape only where it beats v17.

## Oct 29 ~4:15-4:45 AM ET: bf16 v27 sm_90a wgmma swap-AB (branch bf16-wgmma, off origin/moe-v25). Correct, slower; no PR.
Built by tools/build_bf16_v27.py from v17 + tools/bf16_v27_wgmma.cu, regenerated byte-exact at HEAD before each send.
Offline checks on the box (no GPU): NVRTC -arch=sm_90a + ptxas compile (tools/tests/bf16_v27_nvrtc_cc.py: 30/34/42 regs at
NT 8/16/32, no spills) and a CPU emulation of repack + x staging + descriptor addressing + D-fragment epilogue + split-K
under the no-swizzle K-major layout model (tools/tests/bf16_v27_layout_emu.py: exact for M 5-32).
| job | sha | result |
|---|---|---|
| (none) x2 | 1194b4c2… | judge pre-check rejected before any job: "touches harness I/O or secrets" (the cubin went through tempfile / os.fdopen / os.unlink). Fixed: cubin loaded from memory (cupy.cuda.function.Module.load). No quota used |
| 9f218bc6dfb54dce | 5f4e121f… | failed in setup: cupy.cuda.function.Function has no max_dynamic_shared_size_bytes (NVRTC sm_90a compile + in-memory load did work on the judge). Fixed with cupy.cuda.driver.funcSetAttribute |
| ca0ac29de0c347d8 | fcc4113c… | **passed 9/9, 0.927x (0.913-0.941)**: the wgmma kernel is numerically correct on H200. m=8 qkv/o/pooler/unpooler 0.954 / 0.702 / 0.973 / 0.972; m=16 0.659 / 0.555 / 0.664 / 0.582; m=32 0.540 / 0.427 / 0.570 / 0.487 (v17: ~1.0 / 1.08 / 1.07 / 1.03). Other shapes = v17 |
- The layout/descriptor model (LBO = K-direction core-matrix stride, SBO = M/N-direction, swizzle 0) and the D-fragment map
  are confirmed by the judge's correctness tests.
- Why slow (it gets worse as NT grows): each CTA is one warpgroup that fully serializes per 64-k chunk (mbarrier wait -> 4 wgmma ->
  wait_group 0 -> __syncthreads -> refill), stages the whole x slice before the first MMA, and its smem grows with NT
  (qkv NT32 64 KB, pooler/unpooler 96 KB -> 2-3 CTAs/SM). ~1.4 TB/s at m=32.
- Next (if bf16 is revisited): keep the kernel (it is correct) and pipeline it: wait_group(1) with the refill one chunk behind,
  a producer warp for TMA + 2 consumer warpgroups, x via TMA too (or stage per chunk), 8-12 W stages of 8 KB, fewer splits
  with a smaller x footprint. bf16 checks: 19 of 20 used today (v27 x2); 1 left until 8 PM ET.
## Oct 29 ~3:25-3:40 AM ET: attention_decode v15 (branch attn-v15, off moe-v25) -> PR #6
| job | file | speedup (95% CI) | notes |
|---|---|---|---|
| 28ad328cf5284948 | v15a_verify_split.py | 1.232x (1.214-1.251) | exploration: every verify block on the split-KV head kernel. Sliding verify 1.553 / 1.533 / 1.283 (b1/16/32; FA3 was 1.334 / 1.259 / 1.160), full verify b1 1.633 (FA3 1.201), full verify b16/b32 0.691 / 0.666 (FA3 1.033 / 1.014) |
| ab5f387ae32242ba | v15b_verify_route.py | **1.376x (1.355-1.396)** | routes picked from v15a's table by regime: head kernel for sliding verify (any batch) and full verify b1; FA3 for full verify b>=2. Decode unchanged (1.63-1.86). CI clear of v14b's board 1.317 (1.297-1.336) |
- Both carry r6 m2 (dead _qk_norm/_norm_rope_store/_norm_rope_store_torch/_NORMS removed), n3 (_nsplit docstring, combine's
  dead NSPLIT_P2 masking removed + power-of-two assert), n4 (RoPE table index clamped to NPOS - 1). No precision change
  (q E4M3 as before, same kernels as the decode path). Not submitted.
- Next: full verify b>=2 (FA3, ~1.02x) is the only weak cell; our 64-row head kernel is 0.67x there (compute-bound in Triton at
  8 x 8 query rows per KV head). Fusing combine into split / norm+RoPE into the q load were not tried this session.

## Oct 29 ~3:10-3:27 AM ET: MoE session (branch moe-v25, off bf16-tma-wgmma). Nothing clears the 1.40 bar; no PR.
| job | file | speedup (95% CI) | notes |
|---|---|---|---|
| 0d61ab04212b4916 | v25_scale_pair.py | 1.285x (1.266-1.308) | one cvt.e4m3x2 for both rows' scales + integer f16->bf16 re-bias (exact, exhaustively checked) + byte-perm splat. **moe_mma doubled at bs1 (65.5 vs 31.6 us)** and lost 10-20% at bs 8-32: fewer source ops but the SASS is worse (likely __vcmpne2's emulation and/or register pressure/spills; no local NVRTC to read SASS). Dropped |
| 843d33d84aa34786 | v25b_fold_fill.py | 1.391x (1.370-1.412) | v24 kernels + zero-fill launch folded away (route_prep one block per route zeroes its local GEMM-1 row; silu_mul zeroes the local GEMM-2 row) + r4 m1 (.compile() in prepare), m2 (bench stripped), n4 (BLOCK_M by range), n5, n6 (16 B alignment assert), n9 (FP32 noted). bs1 1.809 / bs16 1.391 / bs32 1.117 / v1 1.423 / v16 1.371 / v32 1.322. Within noise of v24 (1.380, 1.359-1.401); bs1 GPU total 39.1 vs 39.4 us (fill kernel gone, route_prep/silu_mul +0.7 us) |
- v25b is the clean candidate carrying all r4 items (minor 3, nearby/skewed shapes, still unmeasured). Same speed as v24
  within noise, so not worth a submit on speed alone; could replace v24 if a reviewer wants the r4 fixes on the board.
- Next ideas (untested): bs32 is issue-bound at ~1.8 TB/s (185 us vs 70.8 us floor). Options: read SASS for the k-step
  (needs NVRTC locally or a check-run dump via cp.RawModule ptx), NB=2 above bs 24 (fewer passes over shared experts),
  fusing silu_mul into GEMM-2 staging again only if it saves the launch at bs1.

## Oct 29 ~2:38-2:50 AM ET: bf16 session 2 (branch bf16-tma-wgmma). Nothing clears the 1.13 bar; no PR.
| check | file | speedup | note |
|---|---|---|---|
| 41a723aa68fe4d79 | v22_router_regime.py | 1.103x (1.087-1.120) | router m<=8 (32,128,4): m=1 0.946x (v7 1.31x), m=8 2.30x (v7 2.5x). Not carried; v21 (same m<=8 config) not sent |
| 42f8408fae1a4a8b | v23_tma_bulk.py | 0.970x | SM90 cp.async.bulk (TMA 1-D) + mbarrier-fed mma.sync GEMV, 1 warp/CTA, 8-16-way split-K. **m=1: qkv_gate 1.072x (first qkv win), sliced unpooler 1.084x**; m>=8 0.46-0.91x |
| 77cfc7faa257400c | v24_tma_route.py | 0.970x | m=1 as v23; m>=2 with 4 warps/CTA + in-CTA reduction (4-16x fewer global atomics): no better (m16 qkv 18.8 us vs 19.4) -> not atomics |
| 1d169b463f4e4dd0 | v25_tma_rt.py | 0.796x | m>=2 with RT=4 rowtiles per warp (x reuse 4x): worse (m32 qkv 64 us) -> the limit is serial mma.sync issue per warp, not x reads |
| cd0ad18a7e114069 | v26_v17_tma_m1.py | **1.124x (1.107-1.141)** | v17 routes + TMA kernel at m=1 on qkv (1.027x this run) and sliced unpooler (1.071x). Same as v17 (1.133x) within noise; CI lower bound 1.107 < 1.13 -> no PR |
Lessons: the TMA bulk-copy feed reaches ~3.8 TB/s on qkv_gate at m=1 (~10 us GPU vs cuBLAS nvjet+splitKreduce ~11 us), but the m=1
gain is 0.4-0.9 us on one shape per load, ~0.3% overall. m>=8 needs many more MMA-issuing warps per SM or wgmma (sm_90a,
NVRTC arch=sm_90a) with x as the B operand from smem; mma.sync from 1-4 warps/CTA is issue-latency bound (one accumulator chain
per (rowtile, n8) pair). Next try: 8 warps/CTA with SS=2/ST=2 (4 KB per warp) and ~35 warps/SM, or a wgmma m64n8k16 swap-AB kernel.

## Oct 29 ~2:30-2:40 AM ET: bf16 session (branch bf16-tma-wgmma, off decode-v13-fused)
Board 1.115x (v6). bf16 checks left at start: 9 of 20. Goal: CI lower bound > ~1.13.
| check | file | speedup | note |
|---|---|---|---|
| 99cd85adb03b464c | v19_large_triton.py | failed timing (9/9 tests) | Triton o/pooler/router at m=128/256. o 0.675x, pooler 0.561x at m=128; router m=128 OOM (smem 294912 > 232448 with BM=128,BN=32,BK=256). Small-m unchanged (~v7). Verdict: naive Triton at verify sizes loses to cuBLAS; do not carry |
Also prepared (committed, not yet checked — okc check / git push were blocked by the safety classifier after v19):
- v20_lib_probe.py / v20b_cute_try.py: discover cutedsl/flashinfer on the judge (CuteDSL TGV is SM100/Blackwell-only; H200 is SM90).
- v21_router_cfg.py: exact-m router configs (exploration).
- v22_router_regime.py: regime-level router retune (m<=8 -> BN=32,BK=128,SPLIT_K=4).
Research note: SGLang's cutedsl_bf16_gemm is Blackwell tcgen05; not usable on H200. qkv_gate still the biggest headroom (wall ~14 us vs SOL 7.9; cuBLASLt nvjet + splitKreduce ~11 us GPU). Hand CUDA GEMV topped out ~3.2 TB/s earlier (v18). Next real attempt needs Hopper TMA+wgmma (SM90), not the SM100 CuteDSL path.
Quota after this session: bf16 checks 8 of 20 left (1 used: v19; the accidental v7 re-send 9e6108c9e3724ee9 was deduped).

## Oct 29 ~2:20-2:30 AM ET: decode_layer v13 (branch decode-v13-fused, off attn-fused)
Built by tools/build_decode_v13.py (reproduces the file byte-exact) from v12_precompile.py + attention v14b.
| check | file | speedup | note |
|---|---|---|---|
| 8ddcc64985b5474e | v13_fused_attn.py | **1.278x (1.259-1.298)** 13/13, score 17,694 us vs v12's 18,135 / 18,158 | v12 + v14's fused norm/RoPE/FP8-q/cache-write kernel (replaces FusedQKNorm + casts + store_cache), FP8 x FP8 Q.K in the split-KV kernels (P.V BF16), FA3 verify path fed the E4M3 q directly. RoPE table read clamped (r6 n4). SMALL_MAX / SMALL_CFG[32] removed; r4 n6 alignment assert, r4 n9 FP32 note |
Per-shape judge speedups (v12 ff2c / v12 b350 -> v13): decode-1 sl 1.416/1.394 -> 1.428; decode-1 full 1.385/1.391 -> 1.422; decode-16 sl 1.291/1.301 -> 1.331; decode-16 full 1.239/1.241 -> 1.322; decode-32 sl 1.074/1.075 -> 1.09; decode-32 full 1.28/1.265 -> 1.266; verify-1 sl 1.361/1.361 -> 1.399; verify-1 full 1.334/1.344 -> 1.368; verify-16 sl 1.195/1.164 -> 1.22; verify-16 full 1.12/1.126 -> 1.162; verify-32 sl 1.167/1.184 -> 1.221; verify-32 full 1.111/1.127 -> 1.128.
Next (decode_layer): decode-32 sliding (1.09x in v13's table) is still the weakest shape; see items 1-3 of the v10 notes below.
SUBMITTED (by the parent, after review): job 093c2ebc301344e8, **1.251x (1.233-1.270)**, 13/13, vs the check's 1.278x
(1.259-1.298). Same bytes (sha256 106b7d87…). Decode shapes reproduce within ±0.01; the gap is all in the verify
16/32 blocks (FA3 path): verify-16 sliding 1.220 -> 1.147, full 1.162 -> 1.111; verify-32 sliding 1.221 -> 1.130,
full 1.128 -> 1.089. The CIs overlap (1.259-1.270), so the check sat at the high end of its range on those shapes.
Quote ~1.25x for v13. The verify-path gain over v12 is smaller than the check suggested.

## Oct 29 ~12:20 AM ET resume (successor worker)
- Branch moe-packed-dequant pushed (was local only). bf16 v8/v9/v10 checks: 1.061x / 0.727x / 0.769x (all
  below the board 1.115x; stream-K and the 12-bit lossless weights lose to cuBLAS) -> bf16 parked.
- MoE v18_gemv (check a428ce02220746c9): 0.577x. CUDA-core grouped split-K GEMV for bs <= 32, Marlin for
  bs 128/256. bs1 66.6 us (Marlin 52.9), bs8 352, bs16 742, bs32 1289 us. Bound by x loads (~1 per weight
  per route: 3-D [route,row,word] tiles put one row per thread) -> LSU-bound, scales with routes.
- MoE v19_gemv_rows (check e99c9700c7754496): 0.220x. 2-D tiles, many rows per thread, one accumulator per
  route: the 16-output dequant asm is materialised for all rows before the FMAs -> 160-255 registers and
  heavy spills (up to ~3000). Even unspilled (32,32,4,4) MB=1: bs1 74.8 us vs Marlin 55.1 us (in-prepare bench).
  Verdict: Triton CUDA-core GEMV can't hold the dequantised weights in registers; with the earlier tensor-core
  tries (v11-v17) MoE-in-Triton is a dead end at 8 checks. Only a hand-written CUDA kernel could compete.
- bf16 v11_cublas_bench (check ae407efff6954988): 1.132x (= v7 path; CI 1.115-1.149). In-prepare bench:
  cuBLAS with pre-transposed weights (torch.mm(x, W.t().contiguous())) is ~1 us faster on pooler (all m) and
  unpooler m16/32; nothing beats SGLang's cuBLAS on qkv_gate; v7's Triton stays best on o.
- bf16 v12_bench_probe (check b8e3f1ba63b044c5): same bench at m=128/256 + probe of in-process JIT toolchains
  (cupy / cuda-python NVRTC / CuTe DSL / tilelang) in the judge image, to see whether a hand-written CUDA
  kernel can be compiled without starting a process (load_inline runs nvcc = a process, not allowed) [CORRECTION, review r4 nit 7: the rules page says native CUDA should be compiled in-file with PyTorch load_inline, so it IS allowed; we use CuPy/NVRTC either way].

## Oct 29 ~1:40-2:00 AM ET: attention_decode fused norm/RoPE/cache store (branch attn-fused, off decode-moe-cuda)
Board 1.117x (v9_regime). Attention checks this session: see the table (window total 1 + these).
| check | file | speedup | note |
|---|---|---|---|
| ed743d60bec04ad8 | v13_fused_store.py | **1.243x (1.224-1.261)** 16/16 | one Triton program per (token, head) does RMSNorm + interleaved RoPE + q out + k/v E4M3 cast + cache-row store. It replaces FusedQKNorm + 2 casts + store_cache (4-5 launches). Decode b1 sliding 16.1 -> 12.8 us (1.58x); verify shapes 1.01-1.25x now (FA3 path gains too) |
| ad626ebbafd94cbb | v14_fp8q.py | **1.325x (1.305-1.345)** 16/16: decode 1.63-1.88x sliding, 1.36-1.66x full; verify 1.02-1.36x | v13 + q written in E4M3 by the fused kernel (FA3 skips its cast) + FP8xFP8 QK dot in the split kernels, PV in BF16 |
Caveat (r6 n2): the Triton 3.9 CPU interpreter's fp32->E4M3 is wrong when rounding carries into the exponent (1.9375 -> 1.0,
0.496 -> 0.25), rounds ties away and gives NaN above ~464; its fp32->bf16 truncates (2.0 itself converts fine). Check cache
bytes locally with a cast-free kernel copy (review tests/t_fused_precast.py pattern), not through the interpreter's casts. GPU tests passed 16/16.
FOR decode_layer: DONE in decode v13 (check 8ddcc64985b5474e, 1.278x): the fused kernel + FP8 q replaced qk_norm + casts + store_cache.
Next (attention): the r2 nits (_nsplit docstring 225 keys, dead NSPLIT_P2 masking) are still open in the attention file (sent files are not edited), so fix them in v15. Fold the split combine into the split kernel (last-arriving program via an atomic counter) to save ~1-2 us per decode shape;
fold the fused norm/RoPE into the split kernel's q load for decode (q_len 1).

## Oct 29 ~1:30-1:45 AM ET: decode_layer with the v24 CUDA experts (branch decode-moe-cuda, off moe-cuda-mma)
MoE v24 was SUBMITTED by the parent after Senior Larper's r4 approval: job d625bb2af63b43c1, 1.380x.
Board decode_layer 1.202x (v6). decode checks this session: 4 (v9-v12); 5/20 in the window.
| check | file | speedup | note |
|---|---|---|---|
| 62ed09f29fd74783 | v9_moe_cuda.py | 1.219x (1.200-1.237) | v24 experts at <=32 tokens + bf16 v17 pooler/unpooler routes. Gains at 1-16 tokens; at 32 agentic tokens moe_mma 127.5 us vs Marlin 124.3, and the cuBLAS-T unpooler lost at 128/256 |
| d761340c8ec04009 | v10_route.py | **1.243x (1.225-1.262)** | CUDA experts at <=16 tokens only, v8 projections, Triton pre-expert RMSNorm (torch's took 6 us) |
| 4e2d567c86e84404 | v11_rmsnorm_all.py | 1.234x | Triton RMSNorm at every token count: within noise, not carried |
| ff2c439e12e54f31 | v12_precompile.py | 1.247x (1.229-1.266) | v10 + r4 m1 (get_function on every kernel in prepare). SUBMITTED as b35042c212354b1f, 1.246x (quote ~1.24x, r5 m1) |
Per shape for v10:
- 1 token: 1.39x; 16 tokens: 1.25-1.31x; verify 8 tokens: 1.35-1.37x.
- 32 tokens: 1.07/1.17x; 128 tokens: 1.15-1.20x; 256 tokens: 1.11-1.18x.
Next (decode_layer):
1. 32 tokens sliding is 1.065x, the weakest shape. The MoE kernel ties Marlin there; it needs NB=2 work items or the issue-bound fixes from the MoE notes.
2. Verify 128/256 tokens use Marlin + FA3. Try v24-style tensor-core experts with NB=4 for 128 tokens.
3. Fuse the CUDA-expert prologue: zero buffer + route_prep + silu_mul + post-norm is about 4 launches.

## Oct 29 ~1:15-1:30 AM ET: bf16 CUDA GEMV session (branch moe-packed-dequant). Nothing beats the board clearly.
Board entry 1.115x (v7). bf16 checks used in this session: 6 (11/20 in the window since 8 PM ET).
| check | file | speedup | note |
|---|---|---|---|
| cdcf4c435dce475a | v13_cuda_gemv.py | failed 7/9 | NVRTC mma.sync GEMV (W repacked into A-fragment order, 16 B x loads); CUDA_CFG keyed by exact m -> KeyError m=7. In-prepare bench data in log |
| ee8f8ee6204d4b78 | v14_route.py | 1.109x | routing picked from the in-prepare bench; the bench overstated CUDA wins (it flushes with a memset, so the judge's cold L2 behaves differently). In the judge, CUDA lost at pooler m1/8/16 and qkv m8 |
| e2894d15903941b8 | v15_cpasync.py | 0.681x | per-warp cp.async ring: 2x slower kernels (cause not found) |
| 693c8d31f9134635 | v16_l2pf.py | 1.009x | v14 + per-warp cp.async.bulk.prefetch.L2: worse everywhere |
| febf8e4adb9c4e1a | v17_best_route.py | **1.133x (1.116-1.150)** | best per-shape routing from judge numbers; about the same as v11's 1.132x, so not clearly above 1.115x |
| 14ba8a134b084788 | v18_dbuf.py | 1.062x | register double buffer (118 regs, loads really in flight) on qkv m<=8 and pooler m<=16. Still worse: qkv m1 15.0 vs 13.7 us, pooler m16 33 us. The CUDA-core-issue GEMV is a dead end at these shapes. |
v17 routing:
- The CUDA GEMV is used only for the sliced unpooler (m<=8), at about 1.02-1.03x.
- Pre-transposed cuBLAS (`torch.mm(x, W.t().contiguous())`) is used for qkv m16/32, pooler m32, unpooler m>=16 and o m128.
- Everything else stays on v7 (Triton for o and pooler at m<=32, SGLang for qkv at m<=8).
Lessons:
- The hand-written CUDA GEMV reaches about 3.2 TB/s (pooler m1 20.7 us kernel); Triton v7 reaches 3.45 and cuBLAS is similar.
- ptxas sinks the batched register loads next to their mmas (38 regs at U=8), so not enough loads are in flight.
- Don't trust the in-prepare bench for small memory-bound kernels; route from judge per-shape tables.
Next steps (bf16):
1. qkv_gate is about 1.00x at every m. The kernel takes 13.5 us against an SOL of 7.9 us, so it has the most headroom. Try a TMA (cp.async.bulk.tensor) or
   wgmma-based swap-AB kernel with a persistent grid of 132 CTAs, plus a deep mbarrier pipeline (4-6 stages of 16-32 KB per SM).
   That is the standard way to reach >4 TB/s; plain loads and cp.async 16 B both topped out at about 3.2 TB/s.
2. (Done in v18; it did not help.) The explicit register double buffer was slower, so the next real attempt is TMA/wgmma (step 1).
3. Use the pooler/o Triton configs at m128/256 (they currently fall back to cuBLAS-T at 1.00-1.02x).

## Oct 29 ~1:30 AM ET: MoE breakthrough (branch moe-packed-dequant) - NOT submitted
Key finding: the judge image has CuPy, cuda-python NVRTC, CuTe DSL, tilelang, quack (probe in bf16 v12,
check b8e3f1ba63b044c5). NVRTC compiles CUDA C++ in-process (no subprocess), so hand-written CUDA is usable.
| check | file | speedup | note |
|---|---|---|---|
| 8b24db54101e43dd | v20_cuda_gemv.py | 0.902x | CUDA-core GEMV via CuPy/NVRTC; bs1 1.697x, larger bs compute-bound |
| 953f0d3cecdd42c1 | v21_mma.py | 1.046x | tensor-core mma.sync m16n8k16 NVFP4 x BF16, A-fragment repack; correct first try |
| ed185d25d2834988 | v22_mma_pf.py | 1.100x | weight prefetch + silu_mul kernel; prefetch depth did not matter (issue-bound) |
| b109e9378bc14d66 | v23_scale_nb.py | **1.325x (1.305-1.345)** | scale path 14 -> 5 ops (cvt chain); NB param; beats board 1.270x (CIs disjoint) |
| d66cab4599544266 | v24_tuned.py | **1.376x (1.356-1.397)** 5/5 | bs1 1.775 bs8 1.398 bs16 1.368 bs32 1.100; v23 + SMALL_CFG from v23's bench (NB=1), BENCH off: the submit candidate |
v23 per shape: bs1 1.753x, bs16 1.232x, bs32 1.029x, bs8 1.350x; bs128/256 stay on Marlin (1.37/1.31x).

Next steps (MoE, in order):
1. DONE: PR #2 (v24) reviewed (r4 APPROVED) and submitted, d625bb2af63b43c1, 1.380x.
2. bs32 is still ~1.0x: kernel is instruction-issue-bound (~26 instrs per k-step). Ideas: lop3-fused dequant,
   amortise the scale cvt across NB, use the tensor-core path for bs128/256 too (Marlin there is ~3-4x off SOL).
3. Zero-buffer memset + silu_mul + route_prep + post-norm = 4 small launches (~8 us at bs1); fuse.
4. Same NVRTC route for bf16 small-m GEMV (qkv_gate/pooler/unpooler), since Triton topped out ~1.03x there.
Tools: /tmp/nvrtc-venv (cuda-bindings + nvrtc cu12) compiles CUDA offline: /tmp/v20/cc2.py <file.cu>.

## MoE Triton rewrite (branch moe-packed-dequant; check-only so far, none beats Marlin yet)
| check | file | speedup | what changed |
|---|---|---|---|
| 16374f95f195436b | v11_triton_packed.py | 0.545× | int64 words of bit-permuted codes, rotate+AND bf16x2 dequant, joins to a [BN,BK] tile |
| 698ae1d5cbd94a3a | v12_splitk.py | 0.510× | + split-K with FP32 atomics for small batches; own FP32 post-norm kernel |
| e57901fe9def4655 | v14_slots.py | 0.578× | dequant straight into the wgmma register A operand (int16 4-slot layout, 11 int ops / 8 codes) |
| 4b804f6c561745bf | v15_tiled.py | 0.723× | tile-contiguous weights/scales; mma.sync (BN=32, 2 warps) avoids wgmma's per-dot wait+fence |
| 603e1fdbeed64cac | v16_persistent.py | 0.722× | persistent grid over routed blocks only; one routing-prep kernel |
| c8a9842f36014fa0 | v17_occupancy.py | 0.725× | occupancy probes: 2 stages, 1-warp CTAs, uncapped persistent grid |

What we learned: cutting the dequant from ~13 to ~2.7 instructions per weight did not by itself help (v11/v12/v14
all ~1 TB/s): the BF16 tile round-tripping through shared memory, then wgmma's per-dot wait + fence, then
scattered 128-byte row reads were the limits.
Outcome (v17, 0.725×; bs1 74.6 / bs16 291 / bs32 359 / bs8 208 / bs128 630 / bs256 845 µs): the board's v9 Marlin
path is faster at every scored shape (52.9 / 153 / 205 / 110 / 332 / 473 µs), so no regime-level hybrid helps either.
The issue/memory utilisation is ~17% / ~21% (latency-bound at ~12 warps/SM, ~168 registers/thread), so closing a
1.4-2.3× gap would need a different kernel structure (hand-written CUDA, Marlin-style), not more tile tuning.
Parked as a dead end after 6 of 20 MoE checks; moved on to the bf16 small-batch GEMV. With those fixed (v15) the kernels reach ~1.2-1.5 TB/s at
~12 warps/SM, still below Marlin (~2 TB/s). The `SWEEP` dicts in v12-v17 are exploration only (one tile config
per scored token count, check runs only); a submit would use one regime-level config.

## Kaggle GPU (cap 15 h for the Kaggle week ending Oct 31 00:00 UTC, shared with Developper)
- Used by this agent: 1 probe session (~3 min, ≈0.05 h) on 2×T4 — confirmed torch 2.11 / Triton 3.6, bf16 `tl.dot` works on T4.
- Verdict: low value. T4 is sm_75 (no FP8, no wgmma/TMA, no H200 timing). CPU `TRITON_INTERPRET=1` tests +
  offline sm_90 compile (`tools/compile_sm90.py`) cover logic and compile errors without spending GPU hours.
  Reserve Kaggle for a future CUDA `load_inline` kernel that needs a real nvcc+GPU correctness run.

## Audit r3 follow-ups
- Done in this branch: STATUS board/ledger refresh (stale running/profile rows filled in from `okc status`), the
  nearest-pow2 paragraph corrected (the count *is* a power of two; m5 not claimed as a measured fix),
  tools/ledger.py now reads sends.tsv, tools/okc_send.py records every attempt (UNPARSED / FILE CHANGED rows),
  tools/localtest_attn.py uses a path relative to itself, one case per line.
- For the next attention / decode_layer version (sent runs/ files are never edited): the `_nsplit` docstring
  must say slices can be shorter than 256 keys (nearest rounding: capacity 1800 -> 8 splits of 225 keys), and
  the combine's `NSPLIT_P2` masking is dead when the count is always a power of two (drop it or say why it stays).

## Previously queued (done)
- attention v9_regime: submitted 18:05 ET, fd13316e3a7f458b, 1.117×. decode_layer v6_attn_regime: submitted
  18:01 ET, af0211f6c18e43eb, 1.202×.

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

## Why full decode splits at batch 1 vs batch >= 2 (audit r2 nit)
Batch 1 has one request: 4 programs/SM already hits the 256-split cap (span // 256 and the 256
limit), and 16/SM would only add combine work; the V-load hoist was only measured at b16/b32. From
batch 2 up, 4/SM leaves the GPU under-filled (b16: 512 programs of ~5,400 keys each), so 16/SM +
hoist (check runs 172c76d6, fd78956d, bb656112: 160/296 us vs 178/349 us at b16/b32).
The split count is still a power of two. Board v9 floors it; audit r2 m5 led to v12 (attention) / v8
(decode_layer), which round to the *nearest* power of two (ties down), so batch 17 gets 128 splits
instead of 64. On the scored shapes this gives the same counts as v9 (checks 86a42f76, 9f1742ce are
redraws of the same launches), and the unscored batches it changes (17-21, 34-43) have not been
measured on H200, so it is not claimed as a fix. v12/v8 are on HOLD per audit r3; nearest-pow2
rides along with the next attention change that has a measured gain.

## Resubmit of MoE 1b5c613200e84954 under Arman's token (Oct 28 ~6:25 PM ET)
`okc submit` of the byte-identical file (sha256 76bb16baff1fe0a6…, via tools/okc_send.py) returned
job cdfb626e97b84cc8 with status **duplicate** (`duplicate_of: 1b5c613200e84954`, same content
hash): the judge did not rerun it, no run was used (8/10 left), and no new board entry exists.

## Job ledger before sends.tsv (frozen; times ET; sha256 = first 16 hex of the runs/ file as committed)

Caveat (audit m2): files before the ledger existed were not hashed at send time; `04269b7477f74417`'s
runs/v4_postnorm_fused.py got a docstring-only edit after that check ran; board job `9cb6d6a409ff4cd6`
was sent after the edit, so it matches the committed bytes. From c89f5f5 on, every file is committed before it is sent.

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
| Oct 28 17:59 | 3795339b0fd94eed | muse_attention_decode | check | v9_regime.py | 3646405dcccf3ac6 | passed | 3,356.8 µs |
| Oct 28 17:59 | e9402d661a8b42b8 | muse_decode_layer | check | v6_attn_regime.py | a297541a0f65e8c6 | passed | 19.01 ms |
| Oct 28 18:01 | af0211f6c18e43eb | muse_decode_layer | SUBMIT | v6_attn_regime.py | a297541a0f65e8c6 | passed 1.202× | 18.83 ms |
| Oct 28 18:01 | c3f668686d154849 | muse_bf16_proj | check | v5_sweepC.py | 745ff209e93e29a2 | passed | 5,112.8 µs |
| Oct 28 18:05 | 50ace93cf34f4af0 | muse_moe_w4a16 | check | v10_bsm2_sweep.py | d5a97e0b36a2c43e | passed 1.167× |  |
| Oct 28 18:05 | e042f303d1fc449c | muse_bf16_proj | check | v6_router_stacked.py | 53c5c18702e1cb65 | passed | 4,280.7 µs |
| Oct 28 18:05 | fd13316e3a7f458b | muse_attention_decode | SUBMIT | v9_regime.py | 3646405dcccf3ac6 | passed 1.117× | 3,372 µs |
| Oct 28 18:06 | f0bd99103bf74d83 | muse_bf16_proj | SUBMIT | v6_router_stacked.py | 53c5c18702e1cb65 | passed 1.115× |  |

## Send ledger (generated by tools/ledger.py from sends.tsv; times ET)

| time | job | problem | kind | file | sha256[:16] (at send) | commit | judge |
|---|---|---|---|---|---|---|---|
| 2026-10-28 18:21:14 | cdfb626e97b84cc8 | muse_moe_w4a16 | submit | v9_postnorm_fused.py | 76bb16baff1fe0a6 | 895ab18 | pending (profile not saved) |
| 2026-10-28 19:43:36 | d1c5ffc5c3eb4cee | muse_moe_w4a16 | submit | v9b_resubmit.py | 387a465e9857920f | 22a1a54 | 1.270x (1.251-1.289), passed=True |
| 2026-10-28 19:44:08 | 1c3e09378cc0443a | muse_bf16_proj | check | v7_audit_nits.py | 95340c7e18ff1893 | 3a9173d | 1.120x (1.103-1.137), passed=True |
| 2026-10-28 19:44:08 | 85e6332651ba4603 | muse_attention_decode | check | v11_smooth_split.py | b832f2f431b7f56f | 3a9173d | 1.112x (1.095-1.128), passed=True |
| 2026-10-28 19:44:08 | e41c3583fe974851 | muse_decode_layer | check | v7_smooth_split.py | c96fc78c5993d0b8 | 3a9173d | 1.187x (1.169-1.204), passed=True |
| 2026-10-28 22:22:28 | 86a42f76b1164863 | muse_attention_decode | check | v12_nearest_pow2.py | f8a23d04bcfa890c | 52ea71e | 1.115x (1.098-1.131), passed=True |
| 2026-10-28 22:22:28 | 9f1742cea00f4460 | muse_decode_layer | check | v8_nearest_pow2.py | ea3b88086cdef7e8 | 52ea71e | 1.193x (1.175-1.211), passed=True |
| 2026-10-28 22:45:11 | 16374f95f195436b | muse_moe_w4a16 | check | v11_triton_packed.py | c0e08e1d71d9766b | e4b1f3d | 0.545x (0.537-0.554), passed=True |
| 2026-10-28 22:54:47 | 698ae1d5cbd94a3a | muse_moe_w4a16 | check | v12_splitk.py | 0da91269b409c323 | 19dd489 | 0.510x (0.503-0.518), passed=True |
| 2026-10-28 23:09:44 | e57901fe9def4655 | muse_moe_w4a16 | check | v14_slots.py | 26e907cd7893c3fe | 4b21850 | 0.578x (0.570-0.587), passed=True |
| 2026-10-28 23:14:56 | 4b804f6c561745bf | muse_moe_w4a16 | check | v15_tiled.py | 5e50df5e78914e6a | 4e3722f | 0.723x (0.712-0.734), passed=True |
| 2026-10-28 23:21:57 | 603e1fdbeed64cac | muse_moe_w4a16 | check | v16_persistent.py | 2287485b0d44be2f | 7f99677 | 0.722x (0.711-0.745), passed=True |
| 2026-10-28 23:28:02 | c8a9842f36014fa0 | muse_moe_w4a16 | check | v17_occupancy.py | 803518baac5982e1 | ef39c26 | 0.725x (0.714-0.736), passed=True |
| 2026-10-28 23:34:11 | 3a7c57476d824e80 | muse_bf16_proj | check | v8_streamk.py | b28ab5344a71cb57 | 7f9e21c | 1.061x (1.045-1.077), passed=True |
| 2026-10-28 23:52:17 | a31e7fe5dd184d13 | muse_bf16_proj | check | v9_zk.py | 59cb50e935a0e206 | cfc51b3 | 0.727x (0.716-0.738), passed=True |
| 2026-10-29 00:08:07 | 633e5dfd0b494aa0 | muse_bf16_proj | check | v10_zk_occ.py | 79202623e5513664 | 6bd49df | 0.769x (0.758-0.781), passed=True |
| 2026-10-29 00:33:05 | a428ce02220746c9 | muse_moe_w4a16 | check | v18_gemv.py | eecc105961109dcd | 4c87f91 | 0.577x (0.568-0.585), passed=True |
| 2026-10-29 00:45:37 | e99c9700c7754496 | muse_moe_w4a16 | check | v19_gemv_rows.py | 1c0d4b62ca9b34a9 | c4a9c23 | 0.220x (0.216-0.223), passed=True |
| 2026-10-29 00:50:26 | ae407efff6954988 | muse_bf16_proj | check | v11_cublas_bench.py | 487fb61c7bb3b01c | a084f61 | 1.132x (1.115-1.149), passed=True |
| 2026-10-29 00:52:00 | b8e3f1ba63b044c5 | muse_bf16_proj | check | v12_bench_probe.py | d6695652c63c6fea | 87356d4 | 1.127x (1.110-1.144), passed=True |
| 2026-10-29 00:56:41 | 8b24db54101e43dd | muse_moe_w4a16 | check | v20_cuda_gemv.py | 74ea274de62746c6 | 229448f | 0.902x (0.888-0.915), passed=True |
| 2026-10-29 01:02:55 | 953f0d3cecdd42c1 | muse_moe_w4a16 | check | v21_mma.py | e3dab931aff74a55 | 0d353d4 | 1.046x (1.030-1.062), passed=True |
| 2026-10-29 01:05:44 | ed185d25d2834988 | muse_moe_w4a16 | check | v22_mma_pf.py | 5f18b181ebebc738 | 7a268e2 | 1.100x (1.084-1.117), passed=True |
| 2026-10-29 01:10:40 | b109e9378bc14d66 | muse_moe_w4a16 | check | v23_scale_nb.py | b851c2463208b408 | edb8de9 | 1.325x (1.305-1.345), passed=True |
| 2026-10-29 01:12:07 | d66cab4599544266 | muse_moe_w4a16 | check | v24_tuned.py | 9cba0e2bb6a650d8 | 628e860 | 1.376x (1.356-1.397), passed=True |
| 2026-10-29 01:16:38 | cdcf4c435dce475a | muse_bf16_proj | check | v13_cuda_gemv.py | 6f48f700322f488c | 03a51f6 | pending (profile not saved) |
| 2026-10-29 01:18:19 | ee8f8ee6204d4b78 | muse_bf16_proj | check | v14_route.py | 45a6b07a07fa7bf7 | a7ee222 | 1.109x (1.092-1.125), passed=True |
| 2026-10-29 01:21:10 | e2894d15903941b8 | muse_bf16_proj | check | v15_cpasync.py | 8f52465c334fec20 | 61a50e9 | 0.681x (0.671-0.691), passed=True |
| 2026-10-29 01:23:08 | 693c8d31f9134635 | muse_bf16_proj | check | v16_l2pf.py | b1f4f0190cc44832 | 515afa4 | 1.009x (0.994-1.024), passed=True |
| 2026-10-29 01:24:37 | febf8e4adb9c4e1a | muse_bf16_proj | check | v17_best_route.py | 9f56573b6e19ab58 | 8d34c13 | 1.133x (1.116-1.150), passed=True |
| 2026-10-29 01:26:41 | 14ba8a134b084788 | muse_bf16_proj | check | v18_dbuf.py | 7b8931a7ca240b66 | e8d5883 | 1.062x (1.046-1.078), passed=True |
| 2026-10-29 01:29:24 | 62ed09f29fd74783 | muse_decode_layer | check | v9_moe_cuda.py | 02e16cae04881ead | d63231f | 1.219x (1.200-1.237), passed=True |
| 2026-10-29 01:32:33 | d761340c8ec04009 | muse_decode_layer | check | v10_route.py | 769b1e5f6b700466 | dd570f2 | 1.243x (1.225-1.262), passed=True |
| 2026-10-29 01:34:10 | d625bb2af63b43c1 | muse_moe_w4a16 | submit | v24_tuned.py | 9cba0e2bb6a650d8 | dd570f2 | 1.380x (1.359-1.401), passed=True |
| 2026-10-29 01:34:57 | 4e2d567c86e84404 | muse_decode_layer | check | v11_rmsnorm_all.py | 4db52119f88eda1e | ea3b527 | 1.234x (1.216-1.253), passed=True |
| 2026-10-29 01:37:27 | ff2c439e12e54f31 | muse_decode_layer | check | v12_precompile.py | b934cc00c842f1c2 | c4c326c | 1.247x (1.229-1.266), passed=True |
| 2026-10-29 01:42:33 | ed743d60bec04ad8 | muse_attention_decode | check | v13_fused_store.py | 946f38729b14889d | a152f37 | 1.243x (1.224-1.261), passed=True |
| 2026-10-29 01:48:32 | ad626ebbafd94cbb | muse_attention_decode | check | v14_fp8q.py | e893f8b43b7cf33a | 4047bb7 | 1.325x (1.305-1.345), passed=True |
| 2026-10-29 01:54:42 | b35042c212354b1f | muse_decode_layer | submit | v12_precompile.py | b934cc00c842f1c2 | f2c065f | 1.246x (1.227-1.264), passed=True |
| 2026-10-29 02:16:03 | efe18d6e4b8c4fe3 | muse_attention_decode | submit | v14b_docfix.py | f9b4b6409ce588ea | 77808d1 | 1.317x (1.297-1.336), passed=True |
| 2026-10-29 02:23:06 | 8ddcc64985b5474e | muse_decode_layer | check | v13_fused_attn.py | 106b7d8773faa722 | 926c76a | 1.278x (1.259-1.298), passed=True |
| 2026-10-29 02:25:43 | 9a588099434d407f | muse_decode_layer | check | v13_fused_attn.py | 106b7d8773faa722 | 7675026 | pending (profile not saved) |
