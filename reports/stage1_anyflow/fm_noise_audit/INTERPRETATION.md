# 固定噪声扫描：误差横跨噪声段，不能归结为缺少低噪声样本

2026-10-08 23:30，FM0 / FM48 两组各54状态完整，新增optimizer updates为0。对应实验B的真实ABot验证集；本扫描使用原片段的自然联合动作，不是新的A/D反事实实验。

## 已核实的结果

| 统计范围 | Original velocity MSE | causal0 | causal48 | FM48相对FM0 |
|---|---:|---:|---:|---|
| 54点等权均值 | 0.173426 | 0.213904 | 0.210760 | 下降1.47%，52/54点下降 |
| 实际8步网格的48点 | 0.130740 | 0.168732 | 0.165915 | 下降1.67% |
| 纯噪声起点，sigma=1 | 0.163484 | 0.276204 | 0.277907 | 均值反而上升0.62%，6点中4点下降 |
| 8步最后一次前向，sigma=0.240781 | 0.244133 | 0.281336 | 0.277918 | 下降1.21% |
| 30步最后一次前向，sigma=0.071108 | 0.514912 | 0.575279 | 0.569519 | 下降1.00% |

每个sigma由2个场景×3个chunk构成，并非6个独立场景。这里只评估GT clean history下的固定状态，没有新增generated-history评测或视频。

1. **低sigma的高raw velocity误差不全是causal引入的。** Original在0.071108也为0.514912；FM48的额外误差为0.054607。该额外误差在sigma=1反而达到0.114423，不能只修低噪声。
2. **多数点的小幅FM拟合改善没有转化为动作和画质通过。** 同状态A/D既有结果：generated delta cosine 0.012540→0.009336；GT为0.017516→0.017641。停车场8/30步也未恢复左右控制。整体MSE不是动作差分方向指标。
3. **低噪声覆盖不足是事实，充分解释失败则不是事实。** 实际192个训练样本仅4个落在sigma≤0.240781；样本占2.08%、Gaussian权重总量占2.06%、加权loss占6.51%。后两项都不等于参数梯度份额。不同训练行的模型、数据和噪声不同，不能当固定输入learning curve。
4. **单次endpoint估计不能与完整采样视频混用。** 本表的endpoint为`z_t-sigma*v`，其MSE严格等于`sigma² * velocity MSE`；低sigma下endpoint误差小，部分是此代数缩放。它不是完整8/30步采样到达GT的距离，也不是额外独立的质量证据。

## 比较的边界

两进程逐状态匹配source/noise/current state/history/GT target/prompt/anchor/audio哈希；Original完整输出哈希也逐点相同。每个checkpoint用自身权重重建history KV；A/D并未在本扫描中变动，所有预测保持KV只读。扫描源码与启动时哈希相同。共216个current/reference前向和8次clean-history前向，未更新参数。

Original仍以相同raw历史、当前状态和匹配条件双向重算历史；student使用causal raw KV。这是明确的结构差异。Original的GT误差还包含条件flow对单个GT样本的误差，不能当绝对可达下界，也不能将student与Original的误差差值直接当latent差分范数。

两进程分别约1193.04/1195.35秒，allocated peak 17383.29/23465.85 MiB，最多两块历史的CPU KV为5403.44 MiB。共享主机、权重offload和两进程显存环境不同；这些只是执行成本记录，不是训练/推理速度对照。完整12 latent rollout最终缓存通常更大。

## 下一实验应控制什么

继续B时，应先把**sigma采样分布与loss权重解耦**，才有可解释的噪声覆盖对照。若比较shift12与shift2.22，保持Gaussian loss weight、初始化、16个训练片段、动作/chunk顺序、噪声seed、trainable bank、LR、batch和有限预算相同；保留sigma=1检查，即使连续随机采样从不恰好取到1。不能同时换mask、anchor或LoRA范围，不能只看总体MSE。

这只是下一受控实验的设计，不是已运行或已证明有效的修复。当前结果不足以选择“大幅低噪声upweight”，也不足以断言改变采样density即可恢复近正交的A/D geometry。验收仍需同时看分sigma误差、同状态当前A/D差分、完整39帧画面及30/8步动作响应；有限预算无改善则停止该因素。

没有新增训练或视频，没有改会议Demo。C/D继续要求可信的局部causal生成与动作能力，不要求Stage1先消除所有长时漂移；目前该前置尚未通过。

[逐sigma结果](RESULTS.md) · [逐状态数据](metrics.csv) · [匹配与分组收据](analysis.json) · [完整执行审计](completion_audit.json)
