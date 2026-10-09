# EXP-003 来源、命令和外部大文件

任务：EXP-003/v1；2026-10-10 04:27–04:54 HKT；GPU 0，AA/AD 顺序两进程。任务发布时 submission HEAD 为 `d536f25a057dc4f5360c87638074c9566dc4e9f0`；根 [next_plan.md](/home/qma/work/GWM/next_plan.md) 是唯一授权。运行期间没有修改 Original 权重、released action LoRA、EXP-002 已验收 endpoint/KV 或旧视频。

`source_manifest.json` 记录并在运行前校验18份冻结依赖：EXP-002 两条 second/third endpoint、`cache_through17.pt`、已发布73帧 `.npy` 和对应逐块JSON，Original/released LoRA、accepted router/interval/config/budget。模型 base shards 与输入 fixture 的更完整 hash 在 [EXP-002 manifest](../EXP-002_native_cached/MANIFEST.md) 和 [EXP-001 source manifest](../EXP-001_v2b_124/source_manifest_v3.json)。本轮 [CPU预检](preflight_cpu.log)实际加载 AA/AD 两份9.632 GB缓存，内部均为50层、chunk索引0/1、token数4680/1950；未用新的prefix重建祖先。

GPU 前的CPU命令：

```bash
.venvs/h3world/bin/python -m pytest -q submission/experiments/EXP-003_native_cached_124/test_contract.py
.venvs/h3world/bin/python -m py_compile submission/experiments/EXP-003_native_cached_124/run_rollout.py submission/experiments/EXP-003_native_cached_124/interval_cached.py submission/experiments/EXP-003_native_cached_124/metered_prefix.py
```

同样11项CPU测试的任务结束后复核输出保存在 [cpu_tests_postrun.log](artifacts/cpu_tests_postrun.log)。GPU命令依次为：

```bash
.venvs/h3world/bin/python -u submission/experiments/EXP-003_native_cached_124/run_rollout.py --config submission/experiments/EXP-003_native_cached_124/config.json --path AA --gpu 0
.venvs/h3world/bin/python -u submission/experiments/EXP-003_native_cached_124/run_rollout.py --config submission/experiments/EXP-003_native_cached_124/config.json --path AD --gpu 0
```

每个进程从自己的EXP-002 cache_through17加载，用原图/动作/时间条件对保存的third5 clean commit一次，随后连续采样三个chunk。90、107帧后仅暂停做视觉审阅，人工输入 `CONTINUE`；未改变模型/参数，review占卡时间仍在任务GPU-hours中。最后124帧不commit。实际阶段、输入hash、行程、cache大小、前向账本和视频 hash 在 [六份逐块JSON](artifacts/metrics/)、[事件日志](artifacts/) 与 [budget.json](artifacts/budget.json)。

并排视频由CPU-only [make_comparisons.py](make_comparisons.py)使用三条真实既有/新生成MP4制作。Original持续A来源是 `H3-World/outputs/2026-10-01-21/action_A_teacher_latents/baseline.mp4`，SHA-256 `b8f96d34edd7342b5b7c2b892e326db52fb61561fa06d4209895ec08914aa1e8`；V2b AA124来源为 [EXP-001已验收原片](../EXP-001_v2b_124/artifacts/videos/AA_rollout_124.mp4)。AD共同73帧并排片直接复制已验收 EXP-002 产物；Original A→D 匹配片不存在。源/目标SHA与标签见 [比较manifest](artifacts/comparison_manifest.json)。

大文件留在 `/home/qma/work/GWM/H3-World/outputs/EXP-003_native_cached_124/`：每条路径 `chunk_22_27.pt`、`chunk_27_32.pt`、`chunk_32_37.pt`、`cache_through22.pt`、`cache_through27.pt`、`cache_through32.pt`、`published_90.npy`、`published_107.npy`、`published_124.npy`。这些文件的SHA逐块记录在JSON并由 [视频/前缀完整性审计](artifacts/video_integrity.json)复核；不会进入Git。小视频和日志的提交副本列在 [artifact_manifest.json](artifact_manifest.json)。

Judge正式验收：[FINAL_REVIEW](judge/FINAL_REVIEW.md)。原始Worker报告/metrics分别冻结为worker_report.md、worker_metrics_snapshot.json；原始逐块JSON/日志不改写。
