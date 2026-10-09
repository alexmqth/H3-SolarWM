# 首块消融说明了什么

**固定权重、固定状态，只改变非action prefix的视频反馈，就足以明显改变A/D velocity差分。当前失配不能仅归咎于历史累计漂移、AnyFlow训练或动作行索引。单边改动又不足以恢复原始几何。**

两个真实验证场景，采用已经保存的causal step00生成endpoint，各取三个sigma，共六个首块状态。这里没有历史KV，所有条件固定，当前5个latent对应的action rows成对换为A/D。Original参照也只读取相同首块范围，使用匹配条件及directed action predicate；这不是完整39帧Original推理。

| 分支 | 整体velocity平均cos | A/D delta平均cos | delta范数比均值 |
|---|---:|---:|---:|
| Original：own action＋prefix读取当前video | 1.000000 | 1.000000 | 1.000000 |
| 仅禁止通用prefix读取video | 0.991841 | −0.061621 | 0.692659 |
| 仅额外开放过去action的直接读取 | 0.999107 | 0.240543 | 0.988708 |
| 两项变化均存在：当前causal首块 | 0.991830 | −0.019933 | 0.730982 |

从当前causal首块出发，单独恢复prefix的视频反馈，六个状态的delta cosine都上升，均值从−0.019933到0.240543；但最高也只有0.570943，不能称作保真。反过来只恢复own-action规则也未解决。二者与多层hidden state交互，不能把这些cosine差相加，或计算“某条边解释了百分之多少失败”。

当前causal分支的动作差分RMS相对整体velocity的平均比例约1.92%，Original约2.80%；响应不是零。共同的外观/场景velocity可以很接近，但动作对应的小分量依然失配。BF16小差分的数值敏感性仍需保留为限制，本实验四个分支统一SDPA调用形状。

两个中sigma状态另用实际部署的split-query cached路径检查：其A/D差分与full-prefix causal分支逐元素相同，相对差分误差为0；空cache零commit/零字节。这使本实验中这两个状态的失配不能归为split-query执行形状差异，但不能将其推广成所有历史状态/后端都数值等价。

Original分支的cos=1是身份正控：两种规则同时回到Original时，首块mask本来就相同。两个场景各A/D一次独立Original重放，共四次RMSE=0。**这不是新训练获得的改善，更不是已修好的causal rollout。**

## 和SolarWM有什么关系

SolarWM H3的`stage1_window_allows`确实规定condition query只读condition，audio query只读condition/audio；SGF也按这种prefix分组执行。因此，静态prefix并非明显违反SolarWM的实现。

但H3-World发布的动作LoRA依赖其directed action predicate，不能由“采用SolarWM的因果规则”推导出原始H3-World动作函数自然保持。本结果表明这一步会改变当前权重下的动作几何；它可能需要适配训练，或经过受控验证的H3-World专用依赖图。没有测试官方SolarWM checkpoint，本实验不评价官方模型是否遇到同样问题。

源码hash及说明见[solarwm_mask_context.json](solarwm_mask_context.json)。主代码`h3_cached.py`中曾将额外的past-action直接入口描述为保留Original历史动作路径，现已修正注释；计算AST不变，正在运行的冻结训练runtime未动，见[说明修订记录](comment_clarification.json)。

## 接下来怎样使用这项证据

1. 先完成既定真实ABot普通FM48和同配置评测。这次只读消融不改变其训练、anchor、步数或预算。
2. 训练后的动作差分复用相同GT及step00 generated states，检查current/history/action-pair/source哈希；不同checkpoint用自己的权重重建KV，各自A/D期间cache固定。
3. 若普通FM仍未形成可信局部生成和action响应，再设计一条保留chunk因果性、只改明确依赖边的对照；至少检查后续chunk、未来内容负对照、KV/recompute语义及完整视频。不能从首块身份正控直接宣布完整方案成立。
4. AnyFlow、Stage2仍需局部生成/动作/有限映射前置验证；本实验没有新optimizer update，没有新增39/124帧视频，也未替换meeting演示。

[完整数据与图](RESULTS.md) · [原始记录](probe.json) · [逐状态CSV](metrics.csv)
