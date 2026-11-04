# bf16 projection redesign: plan (no code yet)

Written Nov 2, 2026 (ET). Problem: `muse_bf16_proj`. Ours: v6_router_stacked 1.115x (f0bd99103bf74d83), rank ~1487.

## Where we stand (from f0bd9910's per-shape table)

| load | SGLang sum | ours | floor sum | ours | floor-limited cap |
|---|---|---|---|---|---|
| decode 1 | 60.0 us | 56.1 | 28.9 | 1.07x | 2.08x |
| decode 16 | 76.3 | 68.0 | 41.4 | 1.12x | 1.84x |
| decode 32 | 77.7 | 70.7 | 41.8 | 1.10x | 1.86x |
| verify 1 (m 8) | 76.1 | 59.1 | 29.0 | 1.29x | 2.62x |
| verify 16 (m 128) | 85.8 | 80.3 | 43.9 | 1.07x | 1.96x |
| verify 32 (m 256) | 103.3 | 97.5 | 51.7 | 1.06x | 2.00x |

Geomean: ours 1.115x; every kernel exactly at the floor would be 2.05x; every kernel at 85% of HBM bandwidth ~1.74x.
The ~70.8 us context is our decode-32 load (70.7 us of summed kernel time vs a 41.8 us floor). Almost all of our
gain is the router (1.3-2.5x: the stacked hi/lo FP32 split); qkv_gate / o / pooler / unpooler are 0.97-1.09x, i.e.
cuBLAS (nvjet) or our Triton swap-AB at cuBLAS speed. Those four layers are ~80% of every load's time.

## What the board implies (public board, Nov 2 ~2:30 PM ET, 1649 entries)

- Top 1.664x, top-50 1.591x, rank 300 1.542x, a dense plateau at 1.54-1.57x (~300 teams), rank 500 1.445x,
  rank 1000 1.216x, and a second cluster at 1.16-1.23x (~370 teams).
- 1.22x cluster: about 60% of the floor-limited cap. That is "router fixed + a decent split-K GEMV on the dense
  layers": roughly our router win plus ~10% on the four dense layers. Getting there from 1.115x needs the dense
  layers at ~1.12x of SGLang, i.e. ~65-70% of HBM bandwidth (qkv_gate ~12.5 us, pooler ~19 us at m 1-32).
- 1.55x plateau: ~76% of the cap, so every dense layer at ~80-85% of HBM bandwidth at every m, including the small
  ones (o: 3.5 us floor; unpooler at m <= 8: 1.75 us floor, where launch/ramp latency dominates). This is a
  well-known recipe, not a trick: a persistent, weight-streaming kernel that keeps ~40+ KB in flight per SM.
- 1.60-1.66x: the same plus small-shape latency work (one kernel per shape, no reduce launch, PDL-style overlap of
  the prologue) and maybe lossless weight compression (allowed by the rules; BF16 weights from a normal-ish
  distribution have low-entropy exponents, so ~12-13 bits/weight is plausible). Treat compression as a stretch goal.

## What we learned already (STATUS.md, v27-v29)

- wgmma swap-AB with split-K FP32 atomics lost (0.915-0.927x): the S x M x N atomics grow with m; atomic-free base
  time was only ~9% better than cuBLAS. Pipelining with all loads issued up front bought nothing visible.
- v29 (plain partial stores + last-CTA reduction) was built, never routed: the serial last-CTA reduction is a tail.
- In-prepare memset-flush benches overstated wins for small kernels (v13/v14); route from judge tables or flush by
  reading a > L2 buffer (decode diagnostics today calibrated within 1-2% that way).

## Design (one kernel family for qkv_gate / o / pooler / unpooler, m 1-256)

1. Layout: repack W in prepare() into CTA-contiguous slabs (each CTA's [n-tile x k-range] as one contiguous byte
   range), so each CTA streams with 1D `cp.async.bulk` (TMA bulk, no tensor map) into a 4-6 stage mbarrier ring.
2. Warp specialization: 1 producer warp (bulk copies) + 2-4 consumer warps doing `mma.sync m16n8k16` swap-AB
   (W rows as A, the m tokens as B: n8 tiles for m <= 8, more n-tiles up to 32; wgmma only for m 128/256).
3. Grid = exactly 132 x (1 or 2) persistent CTAs; work = (n-tile, k-split) chosen per layer so every SM streams the
   same byte count (qkv_gate n 2304: split-K 8-16; o n 8192 k 1024: no split; pooler n 4096 k 8192: split 4-8).
4. Split-K reduction without atomics or a serial tail: thread-block clusters of S CTAs (S = 2-8) sharing an n-tile,
   partial sums reduced through DSMEM into the cluster leader, which writes BF16 once. No workspace, deterministic.
5. Small shapes: unpooler at m <= 8 reads only its 1024-row slice (as SGLang); o and unpooler get a no-split path;
   keep the stacked router as is.
6. Validation plan: CPU layout emulation (as bf16_v27_layout_emu / v29_partials_emu), NVRTC sm_90a compile with
   regs/spills, then one in-prepare diagnostic check that times each layer x m against cuBLAS with read-flush cold
   L2 (the decode diagnostics today showed this calibrates to ~1-2%), then route per (layer, m) only where it wins.

## Expected value and cost

- Reaching the 1.22x cluster (dense layers ~+10%): ~rank 1000 from ~1487. Reaching 1.45-1.55x: ~rank 300-500.
- Overall-board value per +0.05x is lower than decode or MoE (STATUS model: bf16 ~5 places, MoE ~9, decode ~23),
  but bf16 is the problem where we are furthest down its own board, and the same streaming kernel is the core of
  the MoE / decode expert GEMMs (NVFP4 dequant on top), which run at ~2.2-2.9 TB/s today (decode check
  a6f4e782a8fc4db3). Building the streaming kernel once pays in three problems.
- Cost: ~1 day for the kernel + emulation + one diagnostic check, then 1-2 routing checks. No Kaggle needed (Kaggle
  GPUs are not sm_90; the in-prepare judge bench replaces it).
