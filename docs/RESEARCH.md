# Researcher brief (Oct 28), condensed. Source: Researcher agent; links verified by it.
SGLang baseline times: bf16_proj 4.83ms, attention 3.77, MoE 13.45, decode_layer 22.62, prefill_layer 387.40.
Timing: alternating CUDA graphs, cold L2 -> weights from HBM, floor ~ bytes / 4.8 TB/s. Check TASK.md for whether "8-token verify" makes M up to ~256.

What SGLang runs (commit a55ab7d5):
- MoE: moe_align_block_size -> moe_wna16_marlin_gemm (gate/up) -> silu_and_mul -> Marlin down -> post-expert norm + combine. python/sglang/srt/layers/muse_spark_fp4_moe.py
- Attention: FlashInfer TRT-LLM XQA, BF16 Q. layers/attention/trtllm_mha_backend.py, models/muse_spark.py
- Fused ops: python/sglang/kernels/ops/muse_spark_v12 ; BF16 GEMM: kernels/ops/gemm/cutedsl_bf16_gemm.py

1. MoE W4A16 (largest win): fuse gate/up+SiLU (interleave cols in prepare); fuse down GEMM + post-norm + top-k combine + residual. CUDA-core dequant GEMV split-K when ~1-4 tok/expert, Marlin mma.m16n8k16 at larger M. Graph-safe routing (single-CTA align, vllm csrc/moe/moe_align_sum_kernels.cu, or fixed grids over (token,k)). NVFP4 e2m1->bf16 in registers (vllm csrc/quantization/marlin/dequant.h kFE2M1f, or prmt LUT); fold e4m3 group-16 scale x global scale. Refs: vllm marlin_moe_wna16/marlin_template.h, IST-DASLab/marlin, vllm fused_moe.py, Machete, CUTLASS ex55/ex69, TileLang dequantize_gemm & fusedmoe, dropbox/gemlite, vllm marlin_utils_fp4.py.
2. bf16_proj: swap-AB (tokens as N; WGMMA M=64), split-K/stream-K to fill 132 SMs, persistent TMA (Triton tutorial 09-persistent-matmul, DeepGEMM), GEMV for M<=4 (TRT-LLM weightOnlyBatchedGemv), static tuning table keyed on M.
3. attention_decode: split-KV + LSE combine, batch*kvheads*splits >= 132 (FlashDecoding, FA3 flash_fwd_combine_kernel.h); pack GQA heads x 8 verify tokens into MMA M; round Q to FP8 E4M3 (allowed) -> FP8 WGMMA QK^T, beats XQA's BF16 Q; sliding layers fixed bound [pos-2047,pos]; more splits on full layers; flashinfer vec_dtypes.cuh fast_dequant_f8f16x4.
(Brief was truncated after attention section.)

## Rest of brief (Researcher, Oct 28)
Attention refs: flashinfer decode.cuh + attention/hopper, TRT-LLM cpp/kernels/xqa (baseline), FA3 blog/paper 2407.08608, TileLang examples/flash_decoding.
4. decode_layer / prefill_layer: first plug in best MoE/proj/attention, then cut launches (many <10us kernels). Fusions: RMSNorm into next GEMM prologue; QKV GEMM epilogue with QK-norm + RoPE + FP8 KV write; attention -> output norm/gate -> o_proj; MoE down -> post-norm + top-k combine + residual. Check existing sglang kernels/ops/muse_spark_v12 for gaps. Stretch: persistent megakernel (Hazy "No Bubbles", HazyResearch/Megakernels, Mirage MPK). Prefill is compute-bound: FA3-style prefill attention, DeepGEMM/CUTLASS pingpong; smaller relative gains but 387ms baseline.
5. Learning: gpu-mode/reference-kernels problems (nvidia/nvfp4_gemv, amd/moe, amd/mla-decode), kernelbot; RadeonFlow (AMD 2025 winner, split-K FP8 GEMM + MoE); shettysach nvfp4_gemv writeup; ThunderKittens kernels; CuTe DSL examples; Triton 08-grouped-gemm.
Unconfirmed: exact shapes/tolerances; M~256 for verify.
Suggested order: D1 init+profile+roofline+local cold-L2 graph harness; D2-3 MoE then bf16_proj; D4 attention; D5 decode_layer; D6 prefill + tuning.
