# FM → AnyFlow：普通减步与少步训练

`fm2anyflow` 表示从 **Flow Matching（流匹配，FM）** 基础出发，分别测试普通采样减步和 **AnyFlow finite-map（有限时间间隔映射）训练**。它是研究分组名，不表示每个实验都训练了 AnyFlow。

**FM8 = 普通 FM 每个生成块用 8 步采样；FM4 / FM30 分别为 4 / 30 步。AF8 / AF4 = AnyFlow 学生模型每块用 8 / 4 次网络求值。** 数字不是视频帧数或训练更新次数。NFE（Number of Function Evaluations）计去噪采样的网络求值；clean KV commit、编码和解码开销另计。

| 顺序与尝试 | 要解决的问题 | 是否新训练 | 结论与视频入口 |
| --- | --- | --- | --- |
| [1. FM8 续写](fm8_continuation/README.md) / EXP-004 | 原权重直接从 30 步减到 8 步，是否还能续写？ | 否 | 共同 30 步首39帧后续到73帧，有限可行；`fm8_continuation/videos/` |
| [2. 全程 FM8](fm8/README.md) / EXP-006、010 | 首窗也改成8步，并与真实滑窗组合，能否仍工作？ | 否 | 73帧及158帧有限可行，画质/晚切换 PARTIAL；`fm8/videos/` |
| [3. AnyFlow AF8](anyflow/README.md) / EXP-007、012 | 训练能否使少步映射比普通 FM8 更好，并迁移到其他场景？ | 累计32次 finite-map 更新；迁移评估未加训 | 停车场续写有限可行，无整体收益；村落 AA 结构失败，当前 checkpoint 停止扩展；`anyflow/videos/` |
| [4. FM4 / AF4](fm4_vs_af4/README.md) / EXP-009 | 把每块求值进一步降到4次，训练后的 AF 是否更稳？ | 本次评估无新训练，AF复用上述学生 | FM4续写有限可行；AF4 AD第三块结构失败；`fm4_vs_af4/videos/` |
| [5. FM30 / FM8 场景迁移](scene_transfer/README.md) / EXP-011 | 停车场的减步效果能否在工业、村落两图重现？ | 否 | 56帧有限可行；村落FM8颗粒和重影更重；`scene_transfer/videos/` |
| [6. 后续训练的数据准备](data_preparation/README.md) / EXP-013–017 | 教师候选是否动作正确？真实数据与条件协议是否可用？ | 否 | 教师AD并非全合格；真实VAE与CPU布局分别有证据，实际模型回归待做 |

当前收益最明确的是普通 FM8。AnyFlow 的已测 checkpoint 没有显示整体优势，因此保留负结果、停止当前配置扩展。四步续写都借用了 FM8 首窗，不能称为全程四步生成。FM8 与 FM30 自身历史不同的比较，也不能称为同 KV 的纯步数消融。

后续 DMD 训练归入 [V4：SGF / DMD](../../v4/README.md)。[缩写表](../../GLOSSARY.md) · [返回 V3](../README.md)
