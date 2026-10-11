# 数据与输入协议准备：为后续训练检查“答案”和“输入”

这里收集训练前的准备工作。**尚未据这些材料训练新的多场景 AnyFlow 或 DMD 模型。**

| 入口 | 实际做了什么 | 视频是什么 | 解决什么问题 / 当前限制 |
| --- | --- | --- | --- |
| [模型生成的候选监督](teacher_generated_targets/README.md) / EXP-014 | 用冻结 V3 FM30 从4张初图分别生成 AA / AD 56帧 | 模型生成的候选答案，四条并排片 | 检查教师能否提供正确动作监督；s2/s3 的 D 反向未证实，不能把四图全部当合格标签 |
| [真实录屏与 VAE 重建](real_data/README.md) / EXP-015/016 | 提取真实56帧与实际动作，比较同源39/56帧编码及重建 | 真实录屏 / VAE重建并排，四条片 | 确认真正的训练输入及编码前缀一致性；没有真实 D 反事实答案 |
| [固定动作布局 CPU 候选](../../../../experiments/EXP-017_v3_fixed_action_layout/README.md) / EXP-017 | 固定位置模板，隔离未来动作文本长度对当前坐标的影响 | **没有视频**：只做 CPU 元数据和布局检查 | CPU候选通过；真实文本编码器、模型forward、KV数值和生成质量未测 |

`teacher` 是提供监督的较慢参考模型；`target` 是学生以后可能要学习的目标输出。这里的 `teacher_generated_targets` 比旧名 `v3_teacher_targets` 更明确：它是候选数据，不是 V3 的又一个模型版本。GT（Ground Truth）指真实记录，不能用教师生成结果冒充。

[返回 FM → AnyFlow](../README.md) · [缩写表](../../../GLOSSARY.md)
