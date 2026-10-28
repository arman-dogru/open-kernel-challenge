# Kaggle GPU flow

Only the notebook `armandogru/open-kernel-challenge` is used (never any other notebook/dataset on
the account). Budget: at most 15 GPU-hours for the Kaggle week ending Oct 31 00:00 UTC, shared with
another agent — use it only when a correctness check genuinely needs a GPU, batch tests per session.

What's there (probe run, Oct 28): 2x Tesla T4 (sm_75), torch 2.11.0+cu128, triton 3.6.0. BF16 `tl.dot`
compiles and is correct on T4 (emulated); FP8 tensor-core paths and H200 timing are not testable.
Useful for: catching Triton compile errors and logic/correctness of kernels that decode FP8/NVFP4 bits
manually (our kernels do), on real CUDA instead of the CPU interpreter. Not useful for speed.

Push/run (from the box):
```
export KAGGLE_API_TOKEN="$KAGGLE_KEY"
cd /workspace/kaggle-okc            # kernel-metadata.json (id armandogru/open-kernel-challenge, enable_gpu)
/workspace/kaggle-venv/bin/kaggle kernels push -p . --accelerator NvidiaTeslaT4 -t 900
/workspace/kaggle-venv/bin/kaggle kernels status armandogru/open-kernel-challenge
/workspace/kaggle-venv/bin/kaggle kernels output armandogru/open-kernel-challenge -p out
```
Sessions are batch "Save & Run" versions: they stop on their own when the notebook finishes.
