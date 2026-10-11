# V4：SGF 路线下的 DMD 尝试

按研究路线，V4 收集 **SGF（Self-Gradient Forcing，自梯度强制）** 相关研究，当前主要是 **DMD（Distribution Matching Distillation，分布匹配蒸馏）**。SolarWM Stage2 通过 SGF 训练因果学生，在自身生成样本上结合冻结教师和可训练 fake-score critic 来做 DMD。

目标是让少步学生在自己的生成分布与历史条件下训练，缓解训练时干净历史与推理时自身历史之间的差异，并改善少步输出质量。项目目前只完成有限 DMD 尝试，**V4 是研究分类，没有已验收的 V4 生成模型**。

| 尝试 | 实际范围 | 结果 | 入口 |
| --- | --- | --- | --- |
| V4-DMD / EXP-008 | V3 因果协议，学生继承 EXP-007 AF-train-32updates；真实33B三角色，累计8个student cycle；固定C1/current-block on-policy | 训练工程通过；最终AA C2全部17新增帧彩噪，生成FAIL，当前配置归档 | [报告、三列比较与失败原片](dmd/README.md) |
| 早期 Stage2-lite / 旧 Breakthrough03 | 旧 anchor/adapter 协议下的 critic / DMD 工程探索 | 链路可运行；视觉与动作未同时通过 | [旧协议报告与视频](dmd/legacy_stage2_lite/README.md) |

EXP-008 使用冻结 V3 causal teacher；只对固定首块后的当前块做 on-policy 训练，未完成完整 SolarWM Stage2 长程自回归协议复现。早期 Stage2-lite 采用另一套条件与权重，也不能作为 V3/V4 当前协议的成功证据。

旧实验编号、`EXP-008_v3_dmd` 原始目录、checkpoint、视频画面中的旧名保持可追溯；当前展示名称为 **V4-DMD**。8-cycle 是训练次数，DMD8 / 8-map 是每块学生的8次映射求值，两种“8”含义不同。

[返回汇报导航](../README.md) · [V3：FM → AnyFlow](../v3/fm2anyflow/README.md) · [缩写表](../GLOSSARY.md)
