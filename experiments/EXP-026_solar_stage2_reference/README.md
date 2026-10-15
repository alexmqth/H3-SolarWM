# EXP-026/v1 · Official SolarWM H3 Stage2-fix reference preflight

**Status: BLOCKED_ACCESS; CPU preparation complete, official sample inference not run.** This is the SolarWM camera-conditioned reference, not an H3-World text-action A/D evaluation. EXP-025 is already closed and is not rerun here.

The released 33B base and Stage2-fix EMA are present. The candidate [single-case config](candidate_single_case.yaml) parses with the official loader; its exact changes from the [official SP4 config](../../../SolarWM/configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml) are in [OFFICIAL_CONFIG_DIFF.json](OFFICIAL_CONFIG_DIFF.json). The candidate switches SP4/DP2 to SP8/DP1 so `sample_count=1` passes the official complete-DP-wave rule. **This does not make it executable on an L40:** the current official loader moves the complete 61.73 GiB transformer onto one GPU before FSDP wrapping. An L40 has 44.99 GiB total. No GPU was initialized in this task.

The real test index, silence latents, encoder contract and selected latent shard are absent. Hugging Face returned gated 401 on file download; ModelScope returned 403 on dataset-tree access. See [INPUT_MANIFEST.json](INPUT_MANIFEST.json) and [CPU_PREFLIGHT.json](CPU_PREFLIGHT.json). The sample ID, index row, camera trajectory and decoded result remain **unknown**. Neither the official reader nor an MP4 passed, because no authentic input could be read. No file or metric has been fabricated.

Reproduce the available CPU check from the project root:

```bash
PYTHONPATH=SolarWM/src .venvs/solarwm-h3/bin/python submission/experiments/EXP-026_solar_stage2_reference/prepare_cpu.py
.venvs/solarwm-h3/bin/python submission/experiments/EXP-026_solar_stage2_reference/launch_reference.py
```

The first command prints `CPU_CONFIG_PARSE_PASS_INPUT_BLOCKED`; the second is supposed to exit 2 and list all launch blockers. The launcher has an `--execute` switch, but it also checks the task authorization, official sample identity, reader result and load fit. Under v1 it cannot start `torchrun`, even if the switch is supplied. The conditional GPU path and required design fix are documented in [GPU_PROPOSAL.md](GPU_PROPOSAL.md). Actual execution and evidence are in [WORKER_REPORT.md](WORKER_REPORT.md).
