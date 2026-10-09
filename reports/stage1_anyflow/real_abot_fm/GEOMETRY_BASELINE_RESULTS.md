# 固定状态、仅切换当前 A/D：真实场景的零更新 causal 动作差分

**动作的直接入口存在，输出也随 A/D 改变，但与 Original teacher 的差分方向接近不相关。这个问题在真实 GT history 与 generated history 两套状态上都存在。** 这是机制诊断，不是修复结果；真实数据 FM48 仍在训练。

## 实验与结果

两个独立验证场景，每个三个 chunk、三个 sigma：0.939540、0.689441、0.240781。每套 18 个状态，共 36 点。student 为 Original H3 + 发布 action LoRA、零输出新增 adapter，使用部署 causal/persistent CPU KV；没有旧训练 adapter、AnyFlow 或 critic。

在每个状态内部固定 raw history、当前 noisy latent、RGB dual anchor、场景 prompt、video/audio noise、位置编码和时间条件；只把**当前 chunk** 的 action rows 换成 A/D，过去和未来动作保持不变。比较瞬时 FM velocity，不混入 AnyFlow 有限区间平均速度。

| 状态来源 | 点数 | 整体 velocity 平均 cosine | A/D delta 平均 cosine | delta cosine 范围 | student/teacher delta 范数比 |
|---|---:|---:|---:|---:|---:|
| 真实 GT endpoint/history | 18 | 0.988193 | **0.017516** | −0.018145～0.073592 | 0.315～1.364 |
| 固定 step00 generated endpoint/history | 18 | 0.994733 | **0.012540** | −0.258759～0.142183 | 0.346～1.012 |

首块尚无历史：GT 的首块 delta cosine 均值 0.034156；generated endpoint 的首块均值 −0.019933。这排除了“所有失配都必须由历史累积漂移造成”的解释，但没有排除当前 noisy-state 分布、anchor、prefix 依赖图等影响。

GT 与 generated 两套诊断同时改变 endpoint、history 和相应 RGB anchor，不是只改 history 的单变量消融。每个 A/D pair 内的状态控制是严格的；FM48 后还会复用这两套**相同固定状态**做训练前后比较，不混用新模型的自生成状态。

## 检查了什么

- 六个实际 packed layout 中，当前视频读取自己的 action rows、当前 action 读取对应视频的反馈边均保留；时间 spans、global positions 和 anchor 协议已有检查。详细路由差异见下方链接。
- 每套六组历史 KV 内容哈希、commit 计数与参数 version 在干预期间不变。
- 两个场景的 chunk1 分别以当前 A、当前 D 重建历史，历史 KV 哈希相同，当前动作没有回写过去。
- 每套六个中 sigma 状态重复 student 与 teacher 前向，RMSE 全为 0。
- student 与 Original teacher 使用统一 SDPA 后端。teacher 关闭新增适配参数，但保留发布 H3 action LoRA。
- 生成历史由完成的 step00 30-step/chunk 视频对应 latents 取得，source/checkpoint/state/layout 哈希随原始记录保存。

每套实际执行 84 次预测/重复前向、8 次 clean commits。GT 用时 784.78 s，allocated GPU peak 16160.45 MiB；generated 用时 887.61 s，peak 26597.34 MiB。GPU5 generated 探针原启动因可用显存不足而在加载前退出；其失败记录保留，GPU1 成功运行，进程已退出。这些是诊断成本，不是视频推理效率。

## 该怎样解释

整体 velocity 很相似，不能保证动作控制保真：共同的场景/外观分量可以占据大部分向量能量，较小的动作差分仍然可能变向。这里 delta 的幅度并非全为零，因此“完全没读到 action”不符合观测。

同时，符合当前 causal mask 设计不等于 Original 信息流被完整保留。当前 causal 允许视频直接读过去 action，却关闭通用 prefix 读取当前视频，以及当前 prefix 中过去 action 读取历史视频的反馈；Original 还会双向重算历史而 student 使用 clean KV。这些差异在首块或后续块均可改变动作表示。**没有逐边效应消融之前，不能指定唯一错误边或唯一需训练的权重。**

状态是固定 endpoint 的显式加噪插值，不是实际捕获的 solver 中间态。teacher 与 student 历史内部表示/依赖图不同。BF16 小动作差分的后端敏感性仍需牢记，主比较已统一后端并验证重复性；这不等于消除所有数值影响。delta cosine 也不自动转换成像素动作方向或视频质量结论。

实际画面证据另列：零更新 causal 在第二真实场景即使用30步/chunk也在约24帧后人物分解；第一场景相对完整。Original在第一新图的纯A/D自身响应很弱，已保留这个弱正控并补已知停车场正控。模型是否有效必须同时看完整视频、同状态差分与动作正控。

## 证据入口

- [GT 18点及逐chunk/逐sigma统计](geometry/step_00_gt/RESULTS.md) · [CSV](geometry/step_00_gt/metrics.csv)
- [固定 generated 18点及逐chunk/逐sigma统计](geometry/step_00_step00_generated/RESULTS.md) · [CSV](geometry/step_00_step00_generated/metrics.csv)
- [Original 与 cached causal 的路由差异](ATTENTION_ROUTE_INTERPRETATION.md)
- [完整视频与弱正控评审](BASELINE_COMPLETE_REVIEW.md)
- 上一轮统一完整历史重算的实验A、已训练AnyFlow128部署KV诊断，归档在相邻 `field_factorization` 与 `generated_action_geometry128`。三者协议不同，不把各自均值拼成同一个训练学习曲线。

接下来完成既定真实 FM48 和匹配评测。当前没有画质/action PASS，也没有将此诊断当作完整 Stage1 或完整 A–D 目标完成。
