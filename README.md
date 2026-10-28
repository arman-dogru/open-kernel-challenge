# Muse Spark Open Kernel Challenge — arman-dogru

https://openkernelchallenge.dev · kernel track closes Nov 4 11:59 PM PT · serving track Nov 5–19.

Layout
- `<problem>/` — one dir per problem, written by `okc init`: `TASK.md`, `AGENTS.md`, `reference.py`,
  `checks.py`, `baseline.py` (SGLang, the 1.00× to beat), `submission.py` (our current best/WIP kernel).
- `<problem>/runs/vN_*.py` — every file sent to `okc check`/`okc submit`, frozen; `runs/profiles/` the
  judge's per-shape timing + kernel profile for each job (`<file>.<job>.txt/.json/.log`).
- `tools/localtest_*.py` — CPU logic tests with `TRITON_INTERPRET=1` against the problem's reference
  (`/workspace/okc-venv/bin/python tools/localtest_bf16.py`). The interpreter has no bf16/fp8, so these
  run fp32 copies: they check indexing/masking/splits, not precision or speed, and not Triton compile rules.
- `kaggle/` — the Kaggle T4 notebook flow (correctness/compile only).
- `docs/RESEARCH.md` — technique brief. `STATUS.md` — scores, job ids, quotas, Kaggle hours.

Rules we hold to: real speedups only — every key attended, NVFP4/BF16/FP8 kept, BF16 activations where
SGLang keeps them, no work outside the timed call, nothing cached from inputs, CUDA-graph safe. Any run
that might be reward-hacky gets `okc withdraw <job>`.

Workflow: edit `<problem>/submission.py` → local test → copy to `runs/vN_name.py` → `okc check` →
read `runs/profiles/` → `okc submit` the best. Quotas: 10 submits + 20 checks per problem per day,
reset 00:00 UTC (8 PM ET).
