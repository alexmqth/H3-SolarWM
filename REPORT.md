# H3-SolarWM：当前研究状态（2026-10-11）

正式参考为V3 Baseline：严格因果、持久KV、Global RoPE，30步124帧有限可行，画质与连续性PARTIAL。

| 研究方向 | 已做什么 | 结论与入口 |
| --- | --- | --- |
| V3 Sliding Window | Global / Local位置、真实5祖先淘汰 | Global158帧有限可行，当前Local无联合收益；[报告](report/v3/sliding_windows/README.md) |
| V3 普通FM减步 | 8步续写、全程8步、4步续写、158帧及两个其他场景 | FM8优先，有限可行且质量PARTIAL；[报告](report/v3/fm2anyflow/README.md) |
| V3 AnyFlow | 新目标时间学生32次finite-map更新，8/4次求值评估与迁移 | AF8停车场有限可行；AF4及村落迁移存在结构失败，当前配置停止扩展；[报告](report/v3/fm2anyflow/anyflow/README.md) |
| V4 SGF / DMD | 三角色DMD训练8个student cycle及最终视频 | 训练工程通过，生成FAIL；[负结果](report/v4/README.md) |
| 后续数据准备 | 教师候选、真实56帧、VAE重建、固定动作布局CPU候选 | 教师AD并非全合格；CPU布局通过，实际模型回归未测；[说明](report/v3/fm2anyflow/data_preparation/README.md) |

FM8是普通Flow Matching每块8步采样；AF8是AnyFlow学生每块8次网络求值。V4是SGF研究分类，不代表DMD已验收生成能力。实验ID与原始目录保留历史身份。

[视频导航](report/README.md) · [浏览器按路线播放](report/index.html) · [术语表](report/GLOSSARY.md) · [下一阶段草案](report/02_next_steps/README.md)

截至09:00获批任务已结束。下一步建议先做固定动作布局的实际模型回归，再检查真实混合动作条件下FM30/FM8的给定历史局部续写；当前无新GPU/训练任务。遵循ROI和可行性原则，失败配置保留并归档，不为微小收益无限追加实验。
