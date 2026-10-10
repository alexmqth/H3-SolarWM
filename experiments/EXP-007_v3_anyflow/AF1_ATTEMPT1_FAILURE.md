# AF1 GPU attempt 1 — stopped before the first model forward

2026-10-11 HKT, EXP-007/v2 Worker incident record. GPU1 command:

`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-007_v3_anyflow/run_af1.py --gpu 1 > H3-World/outputs/EXP-007_af1_run.log 2>&1`

The process exited in `run_af1.py` while constructing its initial result JSON: `sha(__file__)` passed a string to a helper that expected `Path`. The failure happened before `abot.load_pipeline`, adapter install, checkpoint save, or any 33B model call. The immutable first-attempt ledger at `H3-World/outputs/EXP-007_v3_anyflow_af1/budget.json` records 0 forward, 0 backward, 0 optimizer updates, 0 VAE, 3.539 seconds elapsed on reserved GPU1. There is no `result.json` or model checkpoint from this attempt. The full traceback is in `H3-World/outputs/EXP-007_af1_run.log`.

The helper now normalizes its argument with `Path(path)`; `source_manifest_v2_attempt1.json` retains the failed runner SHA. A new `source_manifest_v2.json` was generated after the one-line fix; `py_compile` and CPU source/shape preflight pass. **No second GPU attempt has been started.** The taskbook says “无自动重试”; an approved rerun must preserve this ledger and use a distinct output/attempt ID, count the first 3.539 seconds against the total approved GPU-time cap, and keep the original one-update limit.
