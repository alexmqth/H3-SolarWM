# EXP-026/v1 command log

All commands ran from `/home/qma/work/GWM` on 2026-10-16 HKT. None initialized a CUDA model.

| Command | Exit | Result |
| --- | ---: | --- |
| `PYTHONPATH=SolarWM/src .venvs/solarwm-h3/bin/python submission/experiments/EXP-026_solar_stage2_reference/prepare_cpu.py` | 0 | `CPU_CONFIG_PARSE_PASS_INPUT_BLOCKED`; candidate/diff/manifest/preflight written |
| `.venvs/solarwm-h3/bin/python submission/experiments/EXP-026_solar_stage2_reference/launch_reference.py` | 2 | Correctly rejected zero GPU authorization, missing official data and pre-FSDP memory fit |
| `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits` | 0 | Eight L40, each 46,068 MiB total; read-only capacity/occupancy query |

The Hugging Face dataset recipe/index download returned HTTP 401 `GatedRepoError`; the ModelScope main and latent dataset tree requests returned HTTP 403 `PermissionDeniedError`. The corresponding provider/revision/target facts are in `INPUT_MANIFEST.json`; no credentials or full error headers are stored here. Downloaded bytes: 0. GPU model forwards: 0.
