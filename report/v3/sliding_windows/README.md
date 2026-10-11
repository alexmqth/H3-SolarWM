# Sliding Window：固定历史窗口与位置编码

**SW = Sliding Window（滑动窗口）**。每块只保留最近 5 个历史 chunks（视频块）的 KV；新块提交后淘汰最旧块，解决历史 video KV 随视频持续增长的问题。这里的 Global / Local 指 **RoPE 位置编码协议**，不是更换文本或图像 embedding 模型。

| 尝试 | 位置与缓存协议 | 目的 | 结果与视频 |
| --- | --- | --- | --- |
| [V3-SW-G：Global](global/README.md) | 保留全局视频时间坐标，真正淘汰旧 KV | 首先检查只改变历史窗口能否延长生成 | 158 帧有限可行、质量 PARTIAL；[A / D 视频](global/videos/G1_A_vs_D_158.mp4) |
| [V3-SW-L：Local](local/README.md) | 可见窗口的视频时间坐标重新映射到局部范围，prefix 保留 Global | 检查局部位置是否缓解长时位置外推 | 动作仍存在，但场景/亮度跳变更多，当前无训练方向归档；[A 对照](local/videos/G1_vs_L1_A_158.mp4)、[D 对照](local/videos/G1_vs_L1_D_158.mp4) |

二者均为 EXP-005 的 30 步、同源 AA124 历史扩展；C7 同历史与噪声，C8 各自闭环。Global 在首次淘汰前复现冻结 V3。Local 重新定位历史 raw K 时，历史隐藏状态仍来自过去的计算，**不等价于按新位置重算全部历史**；不能视为无损优化。

仅历史 video KV 容量已验证有界；158 帧不能证明无限长质量稳定或整个系统内存恒定。全程普通 8 步与 Global 滑窗的组合证据放在 [FM-8NFE / EXP-010](../fm2anyflow/fm8/README.md)。

[缩写表：RoPE、KV、prefix、chunk](../../GLOSSARY.md) · [返回 V3](../README.md)
