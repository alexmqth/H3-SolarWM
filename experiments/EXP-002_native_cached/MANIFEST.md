# EXP-002 来源与运行命令

任务：EXP-002/v1；2026-10-10 04:02–04:17 HKT；GPU 0 单进程顺序运行；`.venvs/h3world/bin/python`。Git HEAD 启动前 `submission` 为 `021f1f064d3fb5d7690707475e00649e8909268b`。Worker 未改历史实验或 checkpoint。

固定权重是 Original H3 base + `H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/checkpoints/H3-World/step-10000.safetensors`，released action LoRA SHA-256 `ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526`。冻结输入/原始代码 hash 由 EXP-001 的 [source_manifest_v3.json](../EXP-001_v2b_124/source_manifest_v3.json) 和旧 `H3-World/outputs/2026-10-09-22/chunk_partition_cb/protocol.json` 约束；首窗 `first12_A.pt` SHA-256 `242a1db06bc3423fe21207af5d2ccf59c9f27f07c9d88f22ce9f77921f6b0eea`。完整大权重不随仓库提交。

执行前检查了 GPU 空闲与上述 hash，先运行：

```bash
.venvs/h3world/bin/python -m pytest -q submission/experiments/EXP-002_native_cached/test_contract.py
PYTHONPATH=/home/qma/work/GWM/H3-World/code:/home/qma/work/GWM/H3-World/DiffSynth-Studio-h3-v2:/home/qma/work/GWM/H3-World/outputs/2026-10-08-18/stage1_real_abot_fm .venvs/h3world/bin/python -m pytest -q submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/test_current_prefix.py
```

四次 GPU 命令，按顺序；每次启动前重新核对空闲卡和累计预算：

```bash
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/run_rollout.py --config submission/experiments/EXP-002_native_cached/config.json --stage second --path AA --gpu 0
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/run_rollout.py --config submission/experiments/EXP-002_native_cached/config.json --stage second --path AD --gpu 0
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/run_rollout.py --config submission/experiments/EXP-002_native_cached/config.json --stage third --path AA --gpu 0
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/run_rollout.py --config submission/experiments/EXP-002_native_cached/config.json --stage third --path AD --gpu 0
```

第一条 AA 在代码增强内存缓存身份和全流程 peak 测量前执行，源码 SHA-256 `07946157070fbf91333d042a4194de4d94c6d7f2fa30204748f52fe31513324d`，见 `run_rollout_aa_at_run.py`。后三条使用增强后的 `run_rollout.py`，每条原始 JSON 均含实际 runner、入口、配置和路由源码 SHA。所有前向调用在调用前记账，包括失败尝试。没有失败/重试，预算见 `artifacts/budget.json`。

并排视频从实际保存的 V2b MP4 与新候选 MP4 经 `make_comparisons.py` 生成：

```bash
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/make_comparisons.py --path AA --frames 56
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/make_comparisons.py --path AD --frames 56
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/make_comparisons.py --path AA --frames 73
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/make_comparisons.py --path AD --frames 73
```

`artifact_manifest.json` 列出已复制小视频/JSON 的 hash。原始大文件位于 `/home/qma/work/GWM/H3-World/outputs/EXP-002_native_cached/`：`first12_A_cache.pt`、各路径的 `cache_through17.pt`、`chunk_*.pt` 与 `published_*.npy`。`artifact_manifest.json` 和逐块 JSON 提供各自 SHA；若要完整重放，需要这些外部文件及上述模型依赖。

Judge接受本轮73帧可行性，见[judge/FINAL_REVIEW.md](judge/FINAL_REVIEW.md)。Worker原始报告/metrics已冻结；Judge源码摘要独立保存，原始运行hash不覆盖。
