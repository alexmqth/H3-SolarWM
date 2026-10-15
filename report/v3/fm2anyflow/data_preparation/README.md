# 数据与输入协议准备：为后续训练检查“答案”和“输入”

这里收集训练前的准备工作。**尚未据这些材料训练新的多场景 AnyFlow 或 DMD 模型。**

| 入口 | 实际做了什么 | 视频是什么 | 解决什么问题 / 当前限制 |
| --- | --- | --- | --- |
| [模型生成的候选监督](teacher_generated_targets/README.md) / EXP-014 | 用冻结 V3 FM-30NFE 从4张初图分别生成 AA / AD 56帧 | 模型生成的候选答案，四条并排片 | 检查教师能否提供正确动作监督；s2/s3 的 D 反向未证实，不能把四图全部当合格标签 |
| [真实录屏与 VAE 重建](real_data/README.md) / EXP-015/016 | 提取真实56帧与实际动作，比较同源39/56帧编码及重建 | 真实录屏 / VAE重建并排，四条片 | 确认真正的训练输入及编码前缀一致性；没有真实 D 反事实答案 |
| [固定动作布局 CPU 候选](../../../../experiments/EXP-017_v3_fixed_action_layout/README.md) / EXP-017 | 固定位置模板，隔离未来动作文本长度对当前坐标的影响 | **没有视频**：只做 CPU 元数据和布局检查 | CPU候选通过；真实文本编码器、模型forward、KV数值和生成质量未测 |
| [真实混合动作 GT-history 续写](mixed_action_validation/README.md) / EXP-019 | 真实文本编码、固定布局回归、四场景前39帧 GT 条件下 FM30/FM8 各续写17帧 | GT/FM30/FM8 三列56帧，只有末17帧新生成 | 局部人物/场景基本可辨；动作语义及自由生成仍未通过验收 |

`teacher` 是提供监督的较慢参考模型；`target` 是学生以后可能要学习的目标输出。这里的 `teacher_generated_targets` 比旧名 `v3_teacher_targets` 更明确：它是候选数据，不是 V3 的又一个模型版本。GT（Ground Truth）指真实记录，不能用教师生成结果冒充。

[返回 FM → AnyFlow](../README.md) · [缩写表](../../../GLOSSARY.md)

- [普通FM真实训练通路与资源测试（EXP-022）](../../../../experiments/EXP-022_v3_fm_training_profile/README.md)：前向/KV PASS，2updates已执行；checkpoint反向重算context有风险，训练梯度协议PASS已撤回，修复归EXP-023。后续[32条/128updates计划（EXP-023）](../../../../experiments/EXP-023_v3_fm_small_data/README.md)已授权。
