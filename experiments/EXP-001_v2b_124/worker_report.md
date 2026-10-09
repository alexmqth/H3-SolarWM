# EXP-001 Worker 执行报告：V2b 持续 A/D 124 帧

报告日期：2026-10-10，Asia/Hong_Kong。Task ID EXP-001，当前 plan_version 3。**Worker Status: completed；Judge Acceptance: pending。** 这表示限定实验和交付物已完成，不表示 V2b 升级为正式高效因果模型。研究负责人的正式验收仍以 [next_plan.md](next_plan.md) 为准。

## 结论与验收范围

在停车场单场景、seed 13、Original H3 + released action LoRA、零新训练的 Same-σ 自生成历史协议下，持续 A 与持续 D 均从已有 56 帧续写至完整 **124 RGB 帧 / 24 fps / 5.17 s**。全部新增块中，A 的水平光流代理保持正向，D 保持负向；人物和场景一直可辨，没有早期方案那种严重人物分解或瞬间换场。对照 Original 的两条真实并排 MP4 已生成并通过完整解码。由此可以支持“此协议在这一个场景/seed 下具备持续 A/D 多窗口生成的可行性”，不能扩展成动作切换、长期泛化或高效严格因果化已经通过。

原 v1/v2 四路径门槛失败仍是事实：DA 在第三块 RGB56–72 输入 A 的水平 flow 为 −0.175，前后半段均负，人物更多向镜头运动；AD/DA 按原停止规则止于 73 帧。v3 按用户 ROI 指示仅继续 AA/DD，既未重标原四路径 PASS，也未追加训练、DA 专项诊断、AnyFlow 或 DMD。

## 历史证据与本轮实际执行

历史 V2b 正控仅到 56 帧第二块，见 [历史实验](submission/experiments/11_causal_12_then5_selfhistory/README.md)。本任务复用其 first12 和 second5 生成 endpoint。v1 在四条路径各生成 [17,22) 第三块，得到 AA/DD/AD/DA 四条 73 帧；v2 因 DA 触发预登记 flow 符号门槛而停止。两版任务书与在途报告已由 Judge 归档。v3 没有改变模型协议，仅将后续能力问题缩到持续 AA/DD：各续 [22,27)、[27,32)、[32,37)，累计 RGB 90、107、124。

模型协议：Original H3 底座 + released action LoRA，Single I0、原生 action/time 与全局 RoPE、原始 directed action routing；首块 12 latent，后续每块 5 latent；每块 30-step native schedule、flow shift 2.22。每个 sigma 将自己已生成的完整可见历史临时加到同一噪声水平，与当前块联合重算，只更新当前 chunk。没有未来动作/视频、GT reset、额外 anchor、新 adapter、persistent hidden KV 或输出平滑。保存的历史 latent 与已发布 RGB 均不回改。该 T2 局部联合重算 **不是** strict chunk-causal masked attention + persistent-KV V3。

运行前核对六份 endpoint、输入、released LoRA 和 13 份 base transformer shard SHA。v1 第三块 first12 与 second5 旧 API 回放 relative RMS 均为 0。v3 对 active path、绝对 09:00 HKT 资源截止、活跃进程 GPU-hours 与恢复不重置预算的 CPU 检查为 3 passed。v1 原代码/配置保留；v3 使用独立 [run_rollout_v3.py](submission/experiments/EXP-001_v2b_124/run_rollout_v3.py)、[config_v3.yaml](submission/experiments/EXP-001_v2b_124/config_v3.yaml)和[来源 manifest](submission/experiments/EXP-001_v2b_124/source_manifest_v3.json)。v1 旧 6/4 卡字段只是运行快照，当前 v3 授权为 09:00 前最多 8 卡、以后最多 3 卡，本轮 v3 实际仅使用 GPU 0/1 两张。

## 结果：动作、画面与边界

下表为每段新 17 RGB 帧固定中心裁剪 Farneback 水平光流均值，单位 px/相邻帧；正负是运动代理而非动作正确率。

| 新增 RGB | AA：持续 A | DD：持续 D | AA/DD 边界灰度 MAD |
| --- | ---: | ---: | ---: |
| 56–72 | +0.347 | −0.879 | 4.51 / 9.41 |
| 73–89 | +1.477 | −0.870 | 16.25 / 7.82 |
| 90–106 | +0.685 | −0.611 | 3.22 / 2.30 |
| 107–123 | +0.987 | −1.260 | 3.07 / 2.41 |

全片 124 帧 Original A/D flow 为 +1.077/−1.602，V2b 为 +1.035/−1.050。完整帧序列、所有新增块接触图与必要原尺寸边界检查中，人物保持单体可辨且停车场结构延续。AA 在 RGB72→73 出现明显转身/位置跳变，不能宣称块边界完美；DD 末段人物贴近左下边缘，有构图漂移。静态逐帧查看由 Worker 完成；Judge 已独立审阅相关完整序列和原尺寸细节。当前视觉判断是可用于可行性展示，达不到与 Original 等质或长期稳定的结论。AD/DA 第三块视频仍保留为动作切换限制的证据。

完整性与可复现性：[CPU 视频审计](submission/experiments/EXP-001_v2b_124/artifacts/video_integrity_audit_v3.json)解码所有 124/73 帧 rollout、每块 17 帧和 2 帧边界片，验证 H.264、24 fps、PTS、帧数、endpoint 文件及逐块已发布 RGB prefix SHA；全部通过。旧 73 帧及每个续写前缀均没有被改写。Original A/D 输入审计核对 initial video/audio noise、anchor、prompt 和全局位置等可比条件；Original 为全片双向、原生联合音视频去噪，V2b 为固定 audio noise 与自身历史协议，因此对照是能力比较，不能把变化归于单一因素。

## 效率与实际预算

Original 两条 124 帧各为全片 30 denoiser forward，已保存运行的完整 wall including shared setup 分别为 A **478.4 s**、D **454.6 s**。V2b 六块需 180 sampling forward/路径并每步重算历史；本轮测得从已保存 73 帧继续到 124 帧的三次独立进程增量 wall 合计 A **1216.8 s**、D **1194.9 s**，不包括旧 73 帧生成时间。两种时间范围不同，不能计算公平速度比，也没有 persistent-KV 加速证据。

EXP-001 全任务实际消耗 **300 sampling + 12 diagnostic = 312 denoiser forward**、18 次 VAE decode、0 optimizer update、3704.85 GPU-seconds（1.0291 GPU-hours）。最高单卡 torch.cuda.max_memory_allocated 为 26392.86 MiB，CPU hidden KV 为 0。首个 GPU 进程 03:10:43、最后退出 03:47:40 HKT；v3 两进程最多同时占 2 卡，均在 09:00 前退出。预算和逐路径 latency/显存原始值见[实验指标](submission/experiments/EXP-001_v2b_124/metrics.json)与[账本快照](submission/experiments/EXP-001_v2b_124/artifacts/raw_json/budget.json)。CPU 模型 offload 按旧协议保留，但未单独精确量化其主机内存峰值，不以此声称内存优势。

## 视频与交付物

- [Original A vs V2b AA，124 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/Original_vs_V2b_A_124.mp4)
- [Original D vs V2b DD，124 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/Original_vs_V2b_D_124.mp4)
- [A/D 依次播放的并排总览，248 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/Original_vs_V2b_AD_overview_248.mp4)
- [完整 AA 原片](submission/experiments/EXP-001_v2b_124/artifacts/videos/AA_rollout_124.mp4)、[完整 DD 原片](submission/experiments/EXP-001_v2b_124/artifacts/videos/DD_rollout_124.mp4)、[AD 73 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/AD_rollout_73.mp4)、[DA 73 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/DA_rollout_73.mp4)
- [视频和计时来源 manifest](submission/experiments/EXP-001_v2b_124/artifacts/comparison_manifest.json)、[实验 README](submission/experiments/EXP-001_v2b_124/README.md)、[权重/代码/输入 MANIFEST](submission/experiments/EXP-001_v2b_124/MANIFEST.md)、[旧 v2 失败指标快照](submission/experiments/EXP-001_v2b_124/metrics_v2_interim.json)

以上两条并排片左 Original、右 V2b，标明同首图/seed/noise、步数和不同计时口径；三条新并排片均通过全片解码、SHA、FPS 与 PTS 检查。原始 large latent、未压缩 RGB 和模型权重保留原位；Git 交付目录复制了主要真实 MP4 与关键 JSON，没有覆盖原始文件。

## 可支持的判断、限制与下一步

本任务证明 V2b Same-σ 局部联合重算可在单场景/seed 保持持续 A/D 响应及可辨人物到 124 帧，同时暴露边界和构图漂移。它没有证明动作切换可靠，更没有证明 strict causal + KV 的高效 V3 已实现。性能上还慢于 Original 的量级，少步 AnyFlow 与 DMD 目前仍是旁支探索，不能把本片归因于它们。后续方向应由 Judge 根据这份完整对照决定；Worker 不自行启动新训练或诊断。

**提交验收：** v3 指定两条路径、完整原片/对比片、指标、源码快照和预算均已交付；Judge Acceptance 仍为 pending。此前 v2 四路径 stop/DA FAIL 独立保留，最终主线版本及 Git 提交由 Judge 维护。
