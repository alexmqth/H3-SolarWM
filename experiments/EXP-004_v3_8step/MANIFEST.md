# EXP-004 来源、执行与大文件索引

执行日期：2026-10-10 HKT。任务书快照为 [taskbook_v1.md](taskbook_v1.md)。环境为项目已有 `.venvs/h3world`，GPU 0 单进程顺序运行。四次运行使用同一 [runner](run_rollout.py) SHA-256 `9f16c673977ec6619f6d8a6205a4030f18dd3146298444797759a7e909e17e24` 和 [config](config.json) SHA-256 `365a618adaa25f39a7271f7eeedf1dcc56ddbd6af603c78282f63c71dfa2a040`；这些 hash 也写在四份逐块 JSON 中。关键源码、首窗 latent、released LoRA 和首窗 raw KV 的路径、resolved path、大小及 SHA 见 [source_manifest.json](source_manifest.json)。仓库中不包含 33B base、released LoRA 或大 KV。

运行前的 CPU 检查确认原生 8-step sigma 为 `[1, 0.9395404663, 0.8694517212, 0.7872340698, 0.6894410400, 0.5711835327, 0.4252873230, 0.2407808990, 0]`，由 `configure_video_schedule(..., steps=8, grid="native", flow_shift=2.22)` 产生，而非截取 30-step sigma。检查还确认 EXP-002 AA/AD 首 39 RGB 逐像素一致、首窗 cache 文件存在、AA 原 `published_56.npy` 的文件 SHA 与 EXP-002 记录一致。`run_rollout.py` 在每次运行时复核首窗 latent、released LoRA、accepted interval 和 cache 摘要，并记录实际条件 SHA。

四次实际 GPU 命令按以下顺序执行；每次运行前检查 GPU 0 空闲，并在 AA/AD 第二块后看全部新增帧才继续第三块：

```bash
.venvs/h3world/bin/python -u submission/experiments/EXP-004_v3_8step/run_rollout.py --stage second --path AA --gpu 0
.venvs/h3world/bin/python -u submission/experiments/EXP-004_v3_8step/run_rollout.py --stage second --path AD --gpu 0
.venvs/h3world/bin/python -u submission/experiments/EXP-004_v3_8step/run_rollout.py --stage third --path AA --gpu 0
.venvs/h3world/bin/python -u submission/experiments/EXP-004_v3_8step/run_rollout.py --stage third --path AD --gpu 0
```

对应原始日志为 `AA_second.log`、`AD_second.log`、`AA_third.log`、`AD_third.log`。第三块入口只加载该路径本轮 8-step `chunk_12_17.pt`，先用原 action / native time / sigma=0 提交一次自身第二块，得到 `cache_through17.pt`，再采样第三块。最后一个 chunk 不 commit。四次运行没有失败或重试。

CPU 制片与核验命令：

```bash
for p in AA AD; do for n in 56 73; do
  .venvs/h3world/bin/python submission/experiments/EXP-004_v3_8step/make_comparisons.py --path "$p" --frames "$n"
done; done
.venvs/h3world/bin/python submission/experiments/EXP-004_v3_8step/finalize_cpu.py
.venvs/h3world/bin/python -m py_compile submission/experiments/EXP-004_v3_8step/run_rollout.py submission/experiments/EXP-004_v3_8step/make_comparisons.py submission/experiments/EXP-004_v3_8step/finalize_cpu.py
```

`finalize_cpu.py` 检查 8 个 MP4 全部 H.264/24fps/预期尺寸与帧数/递增 PTS，四段原始计数与预算、AA/AD 同历史条件摘要、第三块使用各自第二块 endpoint、源视频 SHA，以及 `.npy` 级旧 RGB 前缀不回改；结果见 [核验日志](artifacts/final_cpu_validation.log)。

未复制的大文件位于 `/home/qma/work/GWM/H3-World/outputs/EXP-004_v3_8step/`：每路径 `chunk_12_17.pt`、`chunk_17_22.pt`、`cache_through17.pt`、`published_56.npy`、`published_73.npy`。文件摘要在 `artifacts/metrics/*.json` 和 [metrics.json](metrics.json) 中。共用首窗的 `first12_A.pt` 与 `first12_A_cache.pt` 分别仍在 `submission/experiments/11_causal_12_then5_selfhistory/states/`、`H3-World/outputs/EXP-002_native_cached/`，均未重建。30-step 对照片与指标在 [EXP-002](../EXP-002_native_cached/README.md)。
