# 10：历史噪声/时间协议会影响后续窗口动作响应

这是机制上的部分进展，**不是Stage1或完整多窗口通过**。

问题：Original权重＋native时间＋单I0＋局部T2双向重算，首窗口有动作能力，接入clean历史后仍出现A/D同向和多重手臂。

受控干预：保持attention、Original/released LoRA、12latent、30步、图像/动作/噪声/位置不变，只将历史模型输入改为与当前sigma一致的临时加噪历史及对应video时间。保存的历史只读，不修改过去输出；无optimizer，无新LoRA/anchor，T2依旧逐sigma重算。

结果：固定A历史的当前A/D flow从+1.275624/+1.176016变为+1.645495/−1.546733，修正了当前D方向。固定D历史原本已有正确符号；新协议保持，变为+0.614104/−0.673822。D分支重影明显减少，但D历史当前A的RGB59–68出现明显躯干/多重手臂重影。因此按门槛停止，不执行下一窗口。

- [固定A历史，左clean／右同sigma历史，上A／下D](history_A_clean_vs_noisy.mp4)
- [固定D历史，同样布局](history_D_clean_vs_noisy.mp4)

每个网格前8RGB为相同已知历史末段，之后42RGB是当前窗口；总50帧、24fps。不是自由rollout，不是124f或GT条件。历史来源为Original生成参考。完整静态逐帧与原尺寸细节检查记录保留，不冒称实时播放评审。

[完整报告](../../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/VIDEO_RESULTS.md) · [56次同状态探针](../../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/PROBE_RESULTS.md)。原控制重放、首窗identity、repeat均0，field指标独立CPU重算一致；N/C delta cosine衡量协议差别，不是相对正确teacher的保真度。

说明：这支持“历史条件协议是影响动作信息流的因素”，不支持“唯一问题就是time bug”。噪声和对应时间一起改变，且只有两个固定历史、一个后续窗口的视频。画面门槛未过，不能投入AnyFlow/Stage2或替换会议最终demo。
