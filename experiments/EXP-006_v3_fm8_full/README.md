# EXP-006/v1 — V3 普通 FM8 从首块开始

任务书：[冻结EXP-006任务书](taskbook_v1.md)；执行：[worker 报告](worker_report.md)、[机器指标](worker_metrics.json)、[Judge 验收](judge/final_review.json)。固定 Original H3 + released action LoRA；Single I0、native 时间、current-prefix strict causal、Global RoPE、persistent CPU raw KV、12+5+5 latent、seed13、原生 shift2.22。首块 39 帧由本轮 **8-step** 新生成，没有借用旧 30-step 首块。无训练、AnyFlow 或 DMD。

共享首块、AA 与 AD 两条路径均到 73 RGB 帧。第二块从完全相同的首块与 clean-commit KV 分叉，只改当前 A/D 动作；第三块接各自生成的第二块历史。人物和场景可辨，A/D 光流方向相反，但 AA 第三块有持续透明残影；画质为 PARTIAL，不能称长期稳定或成熟少步模型。关键数值见下表，光流是动作辅助 proxy。

| 块 | 平均水平光流 px | 边界灰度 MAD | 采样秒 | clean commit 秒 | VAE 秒 |
|---|---:|---:|---:|---:|---:|
| 首块 0–12 | +0.430 | — | 47.207 | 8.525 | 3.501 |
| AA 12–17 | +0.846 | 3.658 | 39.827 | 5.828 | 5.252 |
| AD 12–17 | −0.321 | 3.851 | 38.977 | 5.666 | 5.059 |
| AA 17–22 | +0.505 | 4.451 | 50.047 | — | 6.633 |
| AD 17–22 | −0.826 | 4.568 | 41.755 | — | 6.420 |

预算账本：40 sampling + 3 clean commit = 43 forwards、5 VAE；五次独立进程 GPU 占用合计 492.856 秒 = 0.1369 GPUh。GPU0；最高 allocated 26.061 GiB，第三块使用的 CPU raw KV 约 8.971 GiB。该时间包含五次独立加载及序列化，不能直接作为同进程完整 73 帧端到端延迟，也不能与 30-step 片简单换算加速比。

小证据在 [artifacts](artifacts)：原始 39/73 帧 MP4、逐块 JSON、AA/AD 两条与已保存 30-step 的并排视频。并排左侧使用 EXP-002 的**已保存 30-step 首块**，右侧是本轮**新 8-step 首块**，因此从首块起生成 history 不同，是跨历史/跨步数比较，不是固定同状态单因素消融。原始大 latent、KV、RGB 数组、其余未裁剪视频留在 `H3-World/outputs/EXP-006_v3_fm8_full/`，大tensor未复制到Git；运行日志与预算另见artifacts/logs及artifacts/budget.json。

[Judge完整验收](judge/FINAL_REVIEW.md) · [独立审计](judge/completed_audit.json) · [产物清单](artifact_manifest.json)
