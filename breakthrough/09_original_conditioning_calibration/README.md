# Original 条件校准：恢复整窗口正控，不等于 causal 已通过

2026-10-09。解决的问题是**局部架构实验缺少可信的 Original 正控**：此前虽使用 Original 权重和 directed attention，却同时沿用了原型的 clean text/action time 与重复首帧条件，动作效果几乎消失。

[三列 A/D 对照视频](conditioning_window12_AD.mp4)保留全部39帧，三列逐次只恢复一个条件；无训练、同首图/prompt/seed13/noise/30steps。完整窗口为12latent，全部当前窗口动作预先已知。

| 条件 | A flow | D flow | A−D |
|---|---:|---:|---:|
| clean text time＋双首帧 | −0.745426 | −0.767189 | 0.021764 |
| 原生 text time＋双首帧 | −0.000594 | −0.799118 | 0.798524 |
| 原生 text time＋原始单首帧 | +1.250421 | −0.977313 | 2.227735 |

解决方案：先回到H3的原生text/action时间`1−sigma`，再恢复单一I0条件，用连续单变量对照校准参照。全部静态帧中人物和停车场结构保留，A/D运动可区分。SolarWM Stage1采用clean text time是其训练策略，不能说是官方bug；未经适配直接切换该条件则不能默认保留H3动作能力。

尚未解决：5latent分块的局部动作能力、后续历史下的可靠控制、GT局部画面、少步与自由rollout。该视频不是persistent-KV、5latent causal或Stage1通过。audio仍固定noise，与原H3联合去噪有区别；seed13单场景的光流也不能泛化成控制准确率。完整原始指标、源码/输入hash与后续局部状态见[主报告](../../reports/stage1_anyflow/02_causal_diagnostics/local_topology/CONDITIONING_RESULTS.md)。旧失败结果全部保留，会议主demo不替换。
