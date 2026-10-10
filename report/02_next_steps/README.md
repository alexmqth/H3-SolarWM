# 下一步：全程普通FM8优先，AnyFlow独立设计

EXP-005/v2已经收口：SW-G通过158帧有限可行性，质量PARTIAL；Local有动作响应但场景/亮度更不稳定，当前无训练方向归档。正式V3 Baseline继续冻结为124帧参考。

下一优先是V3-FM8从首12-latent窗口开始使用8步，解除EXP-004借用30步首窗的限制。采用冻结Baseline Global/causal/KV协议，先首39帧，再AA/AD各续两块到73帧。拟议43forward/5VAE/≤0.75GPU小时、1卡、0训练；尚未授权GPU。普通缺陷按可行性判断，明显无效则停止，不扫描步数或shift。

V3-AF需新target-time-conditioned student和finite-map训练目标，student用自己的权重构造KV，8NFE与普通FM8匹配比较。旧AnyFlow产物不作新协议完成证据；暂不启动训练或DMD。

[本轮Judge结论](../../experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md) · [后续FM8/AF完整独立设计](../../experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md)
