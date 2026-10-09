# 真实 ABot 验证：零更新 causal 并未稳定保留原始行为

**本页全部 causal 视频来自 step_00，不是正在训练的 FM48。** 两个独立验证 episode 的 Original30、causal GT-history30、causal generated-history30 共六条视频已完成，均为完整 39 帧。conditioning/noise 一致性检查通过。

## 完整画面检查

已查看两场景、两历史条件下的全 39 帧 contact sheet，以及第 0/8/16/24/30/38 帧对应图。这里是静态逐帧检查，不是人工实时播放。

- **户外红房子/树林场景**（`118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140`）：Original 和两条 causal 保留人物到第 38 帧。generated-history 的运动偏弱、后段建筑细节和轨迹漂移，没有第二场景那样的严重人物分解。这是单场景、30 steps/chunk 的观察，不能外推到 8 步或长视频。
- **中世纪村落/持剑人物场景**（`dfec8ed3237860eba14d67c089ecd041_D_1750`）：Original 保持完整；GT-history 人物仍在，但存在拼接跳变。**generated-history 约第 23–24 帧开始出现红色重影，26–33 帧人物明显分解，第 38 帧人物大部消失，背景同时模糊漂移。30 steps/chunk 也没有避免这次失败。**
- GT-history 每块接收真实历史，上一块预测的末状态未必等于下一块的真实条件；17/34 附近的重置/跳变是这种 oracle-history 拼接的限制。H3 temporal VAE 还使用重叠解码，不能把拼接边界尖峰直接解释为自主 rollout 崩溃。

因此，“只要用普通 FM 30 步，未训练 causal 就稳定”不成立。第一场景的相对完整和第二场景的失败必须一起保留。零更新结果只界定桥接训练的起点，不能判定尚未完成的真实数据 FM48 有效或无效。

完整对比视频、全帧图与指标分别在 `report/baseline_complete_gt30/` 和 `report/baseline_complete_generated30/`，每条三列依次为 **Real ABot GT / Original30 / Causal0**。

## 固定 GT 状态的 A/D 反事实

两个场景 × 三个 chunk × 三个 sigma，共 18 点；同一 history/current noisy latent 下只替换当前 chunk 的 A/D，过去和未来动作保持不变。

| 范围 | 整体 velocity 平均 cosine | A/D 差分平均 cosine |
|---|---:|---:|
| 全部 18 点 | 0.988193 | 0.017516 |
| chunk 0，无历史 | 0.988610 | 0.034156 |
| chunk 1 | 0.988204 | 0.013026 |
| chunk 2 | 0.987764 | 0.005365 |

六组 KV/commit/参数不变检查通过；两个场景 chunk1 用 A/D 重建历史的 KV 哈希相同；重复 student/teacher 前向误差均为零。动作差分范数比为 0.315–1.364，不能把弱方向一致性说成模型完全没有动作响应。

这说明差分失配在 GT history、乃至无历史时已出现，不能全部归因于 generated-history distribution shift。低差分 cosine 仍不自动等于视频方向错误，也不能唯一定位到某个权重模块。student 固定历史 KV，teacher 在相同输入范围/anchor/时间/后端下双向重算历史，内部依赖图不同。状态是 endpoint 与噪声的显式插值，不是保存的 solver 中间态。详见 `geometry/step_00_gt/RESULTS.md`。

## 原始 teacher 的纯 A/D 正控也必须检查

在第一个真实场景的同首帧、同静态 scene prompt、同 video/audio noise 上，Original30 的纯 A/D 完整生成得到：

| 动作 | 水平光流均值 |
|---|---:|
| A | +0.744212 |
| D | +0.711019 |
| A−D | +0.033192 |

对应帧也未显示清楚的相反运动。**Original 自己在这个新图上就不是强 A/D 正控**，所以不能直接套用旧停车场的 `A>0,D<0,A−D>1` 门槛再将失败归罪于 causal 模型。保留两条原始视频与弱正控结果；真实 ABot 用于真实视频桥接/画质验证，同时在已验证 Original A−D≈2.023 的停车场上补同设置训练前后回归。

停车场 Original 来自此前 legacy 推理；causal0 与 causal48 统一 h3_fp32，后两者是训练前后的匹配比较，与旧 Original 的比较是整条 pipeline 比较，不是单独精度消融。自然 ABot 片段保留联合按键和镜头运动，不把片段之间的总 flow 差当纯 A/D 控制恢复。

## 资源和下一步

causal30 为每块 30 步，三块共 **90 noisy forwards + 3 clean commits**；Original 是整段 30 forwards。CPU KV 峰值 6484.13 MiB。当前时间是单次共享主机、conditioning 预缓存的记录，不是 warmup 多次均值，不宣称加速。MAD 是帧差/活动量，不是画质。

真实 FM48 继续既定预算，架构/anchor/solver 协议保持。结束后分别比较 GT-history 与 generated-history、停车场 A/D，以及 GT 和固定 step00 generated states 的前后动作差分。只有可信局部视频/动作能力成立，才进入后续 AnyFlow 和 on-policy Stage2；不要求 Stage1 提前解决全部长时漂移。
