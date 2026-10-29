# STATUS — updated Oct 28 2026, ~11:35 PM ET

**Rules acceptance: 2026-10-28 ~6 PM ET, rules re-accepted by Arman personally via `okc login` (on his own
Mac); the box's earlier agent-accepted session was logged out and revoked. Submits from here use his token.**

Board (from `okc leaderboard`, Oct 28 ~11:20 PM ET; all provisional until the organisers' post-contest reruns):
- **Overall kernel board: rank 877** (top-1000 reached at ~7:45 PM ET). Prefill (`muse_prefill_layer`) has its own board.

| problem | best run (board row) | rank | job | file (sha256[:16]) |
|---|---|---|---|---|
| muse_bf16_proj | **1.115×** | 1125 | f0bd99103bf74d83 | v6_router_stacked.py (53c5c18702e1cb65) |
| muse_attention_decode | **1.117×** | 941 | fd13316e3a7f458b | v9_regime.py (3646405dcccf3ac6) |
| muse_moe_w4a16 | **1.270×** | 872 | d1c5ffc5c3eb4cee | v9b_resubmit.py (387a465e9857920f) |
| muse_decode_layer | **1.202×** | 760 | af0211f6c18e43eb | v6_attn_regime.py (a297541a0f65e8c6) |
| muse_prefill_layer | — | — | — | not started |

MoE board reference (Oct 28 late): rank 50 / 300 / 500 / 700 / 800 at 2.37 / 1.97 / 1.74 / 1.38 / 1.31×.

PR #1 (audit-r2-fixes) merged at e4aacf1 after audit r3 (APPROVED). Decisions: bf16 v7 not submitted (same GPU
path as the board row); attention v12 and decode_layer v8 on HOLD (scored launches identical to the board rows).

## Quotas (UTC day Oct 29, i.e. since 8 PM ET Oct 28), from `okc problems`
- Checks: MoE 6 of 20 used (v11, v12, v14, v15, v16, v17 MoE checks); attention 1; decode_layer 1; bf16 0.
- Submits: 0 of 10 used on every problem.

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
  kernel can be compiled without starting a process (load_inline runs nvcc = a process, not allowed).

## Oct 29 ~1:15-1:30 AM ET: bf16 CUDA GEMV session (branch moe-packed-dequant). Nothing beats the board clearly.
Board entry 1.115x (v7). bf16 checks used in this session: 5 (10/20 in the window since 8 PM ET).
| check | file | speedup | note |
|---|---|---|---|
| cdcf4c435dce475a | v13_cuda_gemv.py | failed 7/9 | NVRTC mma.sync GEMV (W repacked into A-fragment order, 16 B x loads); CUDA_CFG keyed by exact m -> KeyError m=7. In-prepare bench data in log |
| ee8f8ee6204d4b78 | v14_route.py | 1.109x | routing picked from the in-prepare bench; the bench overstated CUDA wins (it flushes with a memset, so the judge's cold L2 behaves differently). In the judge, CUDA lost at pooler m1/8/16 and qkv m8 |
| e2894d15903941b8 | v15_cpasync.py | 0.681x | per-warp cp.async ring: 2x slower kernels (cause not found) |
| 693c8d31f9134635 | v16_l2pf.py | 1.009x | v14 + per-warp cp.async.bulk.prefetch.L2: worse everywhere |
| febf8e4adb9c4e1a | v17_best_route.py | **1.133x (1.116-1.150)** | best per-shape routing from judge numbers; about the same as v11's 1.132x, so not clearly above 1.115x |
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
2. To force loads to stay in flight in the current gemv_mma, double-buffer the registers explicitly (load batch i+1 before mma of batch i).
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
1. Senior Larper review of PR #2 (v24), then decide on a submit (Arman's workflow: PR -> review -> joint decision).
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
| 2026-10-28 23:09:44 | e57901fe9def4655 | muse_moe_w4a16 | check | v14_slots.py | 26e907cd7893c3fe | 4b21850 | pending (profile not saved) |
