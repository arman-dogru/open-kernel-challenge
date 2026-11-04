# OKC final-night plan (written Nov 4, 3:30 PM ET; planning only, nothing built or sent)

Window: limits reset at 7 PM ET (40 checks / 20 submits per problem). Kernel track closes Nov 5, 2:59 AM ET. Aim to
have the submit in by ~1 AM ET. Base: main 1d4be90, decode v25 (`muse_decode_layer/runs/v25_ws4_fv.py`).

Board: decode 1.396 (CI 1.375-1.417, submit 49368d8c), attention 1.589, MoE 1.539 (CI upper 1.562), bf16 1.115,
prefill 1.115, overall 956 / 1981. Overall rank follows decode_layer (STATUS, Nov 4 1:00 AM), so every
experiment below is ranked by its decode gain.

## Where decode time goes (submit 49368d8c profile, TASK load weights 46 sliding + 16 full, geomean over 6 loads)

Sensitivity model: remove X us from the profiled kernel times and re-score (script recipe in the appendix):

| change | decode score | gain |
|---|---|---|
| moe_mma -3% / -5% / -8% / -10% on every shape | 1.416 / 1.430 / 1.452 / 1.466 | +1.5 / +2.5 / +4.0 / +5.0% |
| _attn_split_rows (full decode b16 / b32: 123 / 234 us; SOL 63 / 126) -10 / -20 / -30% | 1.405 / 1.415 / 1.425 | +0.7 / +1.4 / +2.1% |
| _attn_split (sliding) -15% | 1.404 | +0.6% |
| attn_fvws (verify full) -10% | 1.401 | +0.3% |
| -1 us on every shape (projection configs) | 1.402 | +0.4% |

- moe_mma is 25-71% of every shape, so it is the only lever that reaches the bar on its own.
- Decode already runs the attention board's newest kernels: the per-shape attention times in the decode profile
  match attention v25e's (78433e53). No older attention or MoE path is left to swap in.
- Bar (Larper's rule): two independent checks of the same file, each with CI lower bound > 1.417. The check CI half-width
  is ~0.021, and v25's real centre is ~1.40 (r19 m1). So each check needs a point estimate of ~1.44 or more: **~+3% real gain**.
  None of the experiments below gets there alone with confidence. The plan stacks them into one candidate.

## Ranked experiments

### E1. MoE: reorder codes at prepare so the unpack has fewer shifts (expected decode +1.5-2.5%)
- **What changes:**
  - Change the prepare-time E2M1 code permutation in the w13 / w2 repack so that one bf16x2 register per 32-bit
    word comes out with a single mask (`a = wq & 0x81C081C0`, no SHF), and a3 stops needing its extra LOP3.
  - Edit moe_mma's unpack to match: 3 SHF + 4 LOP3 per k-step instead of 4 + 5.
  - Same bf16 values, same HMUL2 scale multiply, same MMA operands, so outputs stay bit-identical.
  - Apply it to the decode file (v25 base) and to the MoE board file (v26e base).
- **Why:**
  - The SASS (STATUS, Nov 4 3:00 AM) shows ~23 instructions per k-step, of which ~9 are unpack. This saves 2, ~9% of
    the inner loop issue slots.
  - moe_mma is ALU/issue-bound (the ldmatrix load cut gave nothing), so ~3-6% kernel time is plausible.
  - Lower confidence on T <= 32, where weight bandwidth is the limit.
- **Build:** ~2-2.5 h: the builder script plus repack and unpack edits in both files.
- **Local tests:**
  - (a) NumPy emulation of old vs new repack + unpack over every code value at every nibble position and all scale
    bytes. Requires bit-identical bf16 x scale products.
  - (b) CPU repack round-trip on the judge-shaped weights: unpack(new repack) == unpack(old repack) bitwise.
  - (c) NVRTC compile of every SMALL_CFG and MoE config: 0 spills, regs <= v25's (n3's 32-token GEMM 1 spill must
    not grow), and SASS counts showing SHF / LOP3 down (tools/tests/moe_sass_dump.py).
  - (d) "memory" clobber on any new shared or cp.async asm.
  - (e) The existing decode CPU glue test.
- **Checks:** 1 MoE check + 1 decode check (as part of the stack, E5). Kept as a standalone ablation only if the stack misses.
- **Go / no-go:**
  - Decode: two checks with LB > 1.417.
  - MoE board: two checks with LB > 1.562. That moves only the MoE board; pursue it only if it costs no decode slot.
  - Stop if the SASS doesn't show the instruction drop, or MoE check 1 is < 1.55.
- **Kaggle:** no. T4 / P100 have no cp.async or bf16 mma.sync, so moe_mma can't run there. Test (a) runs on CPU anyway.

### E2. Router hi/lo split at 8 tokens (verify 1) (expected decode +0-0.9%)
- **What changes:** `ROUTER_FP32_MAX_TOKENS` goes 8 -> 1, so T = 8 uses the existing BF16 hi + lo `_proj` router
  (FP32 accumulation), the path already used at 9-32 tokens. TASK.md allows it.
- **Why:**
  - In verify 1 the FP32 `F.linear` router shows two sm80 FP32 GEMMs, 12.5 + 9.5 us (decode 1 / T = 1 uses a 7 us gemv).
  - It runs on the side stream next to the pooler (~13 us). The excess is on the critical path: up to ~10 us of
    171 / 185 us per verify-1 shape.
  - The gain is uncertain because of that overlap, and the judge's timing will show it.
- **Build:** 15 min.
- **Local tests:**
  - CPU emulation on generate_input-style data, T = 8, many seeds: hi + lo logits vs FP32. Every expert's
    sigmoid + bias must be within 0.05 of the reference's 8th, plus the 0.03-margin top-8 rule.
  - Glue test.
- **Checks:** part of the E5 stack. Ablation only if the stack misses.
- **Go / no-go:** drop it if the emulation ever breaks the top-8 rule, or an ablation shows verify 1 slower.
- **Kaggle:** no (CPU is enough).

### E3. Full-decode attention split-rows retune (b16 / b32 full, expected decode +0.3-1.4%; also attention board)
- **What changes:**
  - One diagnostic check (attention problem, no decode slot) sweeping `_attn_split_rows` launch configs in an untimed
    prepare block, like v25e1_bench / v26p: split count, rows per CTA, num_warps, num_stages, occupancy for b16 / b32,
    ctx 80000, q_len 1.
  - Then fold the winner into the decode stack.
- **Why:** these shapes run at ~52% of SOL (123 / 234 vs 63 / 126 us). Attention v24 only moved occ 16 -> 4. A 10-20% cut
  is +0.7-1.4% decode and lifts the attention board's decode 16 / 32 full cells.
  A new warp-specialized q_len-1 kernel (the attn_fvws pattern) could win more but is >4 h of work; **not in scope tonight**.
- **Build:** ~1 h for the diagnostic sweep, ~20 min to fold in.
- **Local tests:** the attention CPU compare (host path) and the decode glue test. The rows kernel is Triton, so test
  configs with TRITON_INTERPRET on small contexts for bit-exactness at equal splits (split count changes rounding, so
  re-verify the tolerance case set).
- **Checks:** 1 attention diagnostic, then part of E5. If it wins >= 5% alone, 1-2 attention-board checks
  (LB > attention board UB).
- **Go / no-go:** fold it in only if the diagnostic shows >= 5% on both b16 and b32 full. Otherwise drop it.
- **Kaggle:** no (Triton tuning on T4 says nothing about H100).

### E4. Small projection / config stack (expected decode +0.3-0.5%)
- **What changes:**
  - Pooler m <= 32 -> Triton (64, 256, 1, 4, 4) (v26p diagnostic: -0.5 to -1.3 us).
  - Unpooler m 16 / 32 -> best Triton (~-0.5 us).
  - The 32-token GEMM 1 config that spills 12 / 16 B under MINB 4 (r19 n3): pick the nearest non-spilling one,
    chosen on NVRTC evidence only.
  - #25 / #26 nits in the new source: dead NB 4 MINB branch, comments, cp.async "memory" clobbers.
- **Why:** each is measured (v26p 7140d1f5) or zero-risk. Together they are about the size of the noise, so they
  ride along and are not checked on their own.
- **Build:** ~45 min.
- **Local tests:** the glue test, NVRTC compile / spills for the changed config, and Triton interpret correctness
  of the new pooler / unpooler configs vs torch.
- **Checks:** none on their own (E5).
- **Go / no-go:** drop any item whose local test fails.
- **Kaggle:** no.

### E5. The stacked candidate: v27 = v25 + E1 + E2 + E4 (+ E3 if its diagnostic clears) (expected +2.5-4%, point ~1.43-1.45)
- This is the candidate the bar is applied to. Estimated chance that both checks clear 1.417: **~25-35%**.
- **If check 1 misses:**
  - Don't re-roll the same file (Larper's rule).
  - Send ablations: v25 + E1 alone, and v25 + E2 + E4. Use them to find which part helps.
  - Rebuild with only the positive parts, and only if that is still projected to clear.
- **Check budget:** at most ~10 decode, 3 MoE, 2 attention of the 40 each.

## Timeline (ET)

| time | step |
|---|---|
| 3:30-6:45 PM (if approved now; local only, no checks) | Build E1 (decode + MoE files), E2, E4; all local tests; E3 diagnostic file ready. Commit and push after each. |
| 7:00 PM (reset) | Send in parallel: decode check 1 of v27 (E1+E2+E4), MoE check 1 of moe v29 (E1), attention diagnostic (E3). |
| ~7:30-7:50 PM | Results (queue ~15-25 min wait + ~10 min run). |
| if decode check 1 LB > 1.417 | Decode check 2 of the same file right away; result ~8:30 PM. If both clear: PR with job ids + sha256, report for Larper by ~8:45 PM. |
| if it misses | ~8:00 PM: two ablation checks (E1 only; E2+E4). ~9:30 PM: rebuilt v27b (+ E3 if it cleared). Check 1 by 9:45, check 2 by 10:30 PM. |
| 10:45 PM | **Last first-check send.** After this, no new candidates. |
| 11:30 PM | **Last PR to Larper.** Without two clears by then, v25 stays as the final decode entry. |
| ~12:00-1:00 AM | Larper clears one submit by sha. `okc_send.py submit` once; `okc status --wait`. Records branch, re-fetch boards. |
| 1:00-2:59 AM | Buffer for the judge queue (the backlog has reached ~1600 jobs at peak). No new work. |

MoE board (only if E1's MoE check 1 clears 1.562): check 2 ~8:15 PM, PR alongside decode. It needs its own submit
clearance; it does not move the overall rank.

## Appendix: sensitivity recipe
Read the 12 rows and per-kernel times from `muse_decode_layer/runs/profiles/v25_ws4_fv.49368d8c2c034751.txt`.
Subtract the kernel-time cut from "yours" per shape. Load time = 46 x sliding + 16 x full. Score = geomean over the 6
loads of sglang / yours. Base reproduces 1.3960.
