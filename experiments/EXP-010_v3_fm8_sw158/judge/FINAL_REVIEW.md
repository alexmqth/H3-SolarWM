# EXP-010/v1 Judge 最终验收

2026-10-11 05:39 HKT。**accepted：从首窗开始的普通 FM8 + Global 滑窗，A 继续 / D 晚切换各 158 帧有限可行性通过；画质 PARTIAL，首个晚切换块响应 PARTIAL。** 本任务完成，停止扩展，不加 C9、不调参。正式 30-step V3 Baseline 与 EXP-005 SW-G 保持冻结。

## 真实执行与独立核查

复用 EXP-006 自己的全 FM8 AA73 和 KV，提交已有 C3，再生成 C4–C6 至共同 AA124，随后 C7/C8 A 继续与 D 切换。新增 **62 forward = 56 sampling + 6 clean commit、7 VAE、0 update**；含进程加载/保存的账本累计 **1137.108658 GPU 秒 = 0.315863516 GPUh**，低于 .75 GPUh；单 GPU0 allocated 峰 **26.595658 GiB**。前 73 帧成本属于 EXP-006，不重复计作本轮新计算。

[独立 CPU 审计](branch_audit.json) PASS：36 项冻结来源、实际 native sigma/noise/prompt/history、端点与父缓存 SHA、各段全部 MP4 解码/24fps/递增 PTS、历史 RGB 逐值不变。实际加载 C6 及 A/D C7 缓存张量，全部 50 层核对精确 indices：C6 提交后 1–5，C7 提交后 2–6，C8 读取各自分支的 2–6。真实历史 video KV 为 **14,164,800,000 bytes（13.19 GiB）**，已发生两次淘汰，不再随块数增长；prefix、完整 latent/RGB 和 VAE 解码成本仍可增长。

## 视频验收

Judge 检查了 C4–C6 全部 51 新帧、两路 C7/C8 全部 68 新帧及 C8 原分辨率末帧，并查看 30-step / FM8 并排关键帧。完整逐块记录见 [visual_notes.json](visual_notes.json)。

| 路径 | C7 水平 flow | C8 水平 flow | 观察与判定 |
| --- | ---: | ---: | --- |
| A 继续 | +0.43882 | +0.58664 | A 运动保留，人物/车库可辨；持续透明腿部拖影、视角和几何边界变化，quality PARTIAL |
| D 晚切换 | +0.03414 | −0.85809 | C7 姿态改变但水平响应弱/模糊；C8 反向移动更清楚。切换响应有延迟，首切换块 PARTIAL；主体/场景可用 |

光流为辅助证据。没有持续主体消失或全帧彩噪，普通视觉缺陷按可行性尺度接受；不声称立即、精确或成熟的动作控制。

| 块 | sampling 秒 | commit 秒 | decode 秒 |
| --- | ---: | ---: | ---: |
| C4 | 45.69 | 6.59 | 8.03 |
| C5 | 48.83 | 7.34 | 9.70 |
| C6 | 52.86 | 7.56 | 10.93 |
| A C7 | 47.83 | 7.09 | 11.81 |
| A C8 | 48.96 | 0 | 13.33 |
| D C7 | 48.54 | 7.18 | 11.81 |
| D C8 | 60.41 | 0 | 13.60 |

已有 C3 clean commit 为 13.01 秒；完整阶段/内存见审计。不同块有重复加载和缓存 I/O，不将这些数字表述为连续进程的公平 E2E 加速。

## 交付与结论边界

- [A 158 原片](../artifacts/branch/A/rollout_158.mp4) · [D 晚切换 158 原片](../artifacts/branch/D/rollout_158.mp4)
- [30-step SW-G / FM8，A](../artifacts/comparisons/A_SWG30_vs_FM8_SWG8_158.mp4) · [D](../artifacts/comparisons/D_SWG30_vs_FM8_SWG8_158.mp4)
- 两条并排片独立核对 SHA、158 帧、24fps、1248×424、完整解码与递增 PTS。

比较使用相同原始输入/动作/噪声协议，但首窗与后续生成历史不同，属于闭环方案比较，不是同 raw KV 的单状态消融。本轮零新训练、未使用 AnyFlow 或 DMD。只覆盖停车场 seed13、158 帧约 6.58 秒、C7 才切 D；不证明早期 AD 长时稳定、无限长度或泛化。冻结 long47 fixture 保留 V3 原位置，不等价于默认 47-latent packed 重建。

普通 FM8 已值得保留为低成本候选，当前证据不足以取代正式 30-step 参考。下一项优先检查两个固定非停车场初图的 FM30 / FM8 短程迁移，先做原生 Single I0 输入准备与 CPU 审查；不继续挽救当前失败 AF4/DMD 配置。
