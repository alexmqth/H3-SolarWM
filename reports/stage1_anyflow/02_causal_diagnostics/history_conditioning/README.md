# E1 历史条件对照：方向改善，视觉门槛仍未过

2026-10-09 06:30：E1历史条件对照已全部结束。窗口1两history均恢复A正/D负，但D历史的A分支RGB59–68仍有明显躯干/多重手臂重影；画面门槛未过，按协议停止，不跑窗口2。56诊断＋120采样前向，零optimizer。下一步准备真实动作后果监督的数据/梯度审计，不进入AnyFlow/Stage2。

[完整视频结果](VIDEO_RESULTS.md) · [同状态探针](PROBE_RESULTS.md) · [E2数据准备](E2_REAL_TRANSITION_PREPARATION.md)。

可播放网格：[固定A历史](window1_A/CN_AD_context.mp4)、[固定D历史](window1_D/CN_AD_context.mp4)。左C clean history、右N same-sigma临时加噪；上A下D。前8帧相同已知历史，后42帧当前窗口；总50帧，不是自由生成。

4项实际tiny-H3 CPU检查、56次真实H3探针、120次视频采样均完成，零optimizer。C/N首窗identity、旧C重放与重复误差0；独立CPU重算field指标一致。两份history的A>0/D<0成立；修正的是A历史当前D的符号，D历史原本符号已正确。但D历史当前A在59–68帧有明显多重手臂/躯干重影，故停止本因素，不执行第三窗口。

[冻结协议](protocol.json) · [CPU收据](cpu_receipt.json) · [人工评审](window1_review.json)。源码/305文件runtime/输入hash保持；新实现只用于隔离实验，没有切换生产默认。T2每sigma重算，CPU hiddenKV0，不能声称缓存加速。

复验需提供原基础权重、source_coarse数据与冻结runtime。提交包不包含latent/conditioning/field tensors；完整张量保留在项目outputs。当前仍在研究局部能力，尚未完成Stage1。
