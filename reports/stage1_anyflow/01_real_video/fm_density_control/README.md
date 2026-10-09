# B补充：只改变sigma采样分布的真实视频FM对照

2026-10-09 02:57：全部评测与归档已完成，动作/画面门槛FAIL。停车场30/8步 A−D 为−0.031684/0.073371，A均为负；原GPU/CPU队列已退出。[最终收尾](FINAL_RESULTS.md)。旧运行日志保留为历史，不再扩展本分支。

| 因素 | 旧对照 | 新分支 |
|---|---|---|
| 训练sigma采样shift | 12 | 2.22 |
| Gaussian权重函数的grid shift | 12 | 12 |
| 验证采样 / 权重 / 推理shift | 2.22 / 2.22 / 2.22 | 相同 |
| 真实视频、动作、chunk课程、noise随机序列 | 16train/8validation、39RGB、5+5+2 | 相同 |
| 初始化、rank8 QKVO/FFN/refiner、RGB dual、CPU KV、全历史梯度 | Original＋发布action LoRA；zero-output bank | 相同 |
| budget / batch / LR / seed | 48 / 4 / 3e-5 / 13 | 相同 |

固定的是权重关于sigma的函数。样本sigma变化时，其标量权重和总梯度仍会变化；没有importance correction，不声称两组优化目标完全相同。此次干预正是sample density，不是同时换一套Gaussian normalization。

## 已验证与尚未验证

- [预注册协议](protocol.json)：新预算48封顶，保存0/1/3/16/32/48；无自动扩训或架构变化。
- [CPU前检](preflight.json)：新入口默认行为与旧冻结trainer在FM/AnyFlow的完整历史loss、所有参数梯度及RNG逐元素相同；旧checkpoint仍可精确resume，更改loss policy被拒绝。
- 同一随机序列重建旧192个sigma及末尾RNG状态完全匹配；新分支低/中/高段样本为30/76/86，旧对照4/35/153。历史运行没有直接保存noise tensor hash，重建哈希与当前stream检查的证据层级已明示。
- [45项相关测试](tests.json)通过；[真实GPU stream检查](stream_audit.json)已通过step00参数/optimizer/数据/RNG一致性，初始4片段validation完整记录也逐项等于旧对照。
- [启动记录](launch.json)：GPU1，PID1563900，start ticks255332670，23:47启动。查进程需同时核对start ticks，不能只凭本页或JSON状态。
- [后续评测队列](post48_queue.json)已取得完整48-update收据和审计，当前评测运行中；最多GPU1/0两路，失败不自动重试。队列PID1615283，start ticks255366821。

完整计划：54点noise及GT/固定step00-generated各18点几何已经完成；2自然场景GT30/generated30共4视频已评审。剩余generated8和停车场A/D30/8继续运行，全部要求39帧。teacher和step00基线不重复生成。

`probe_real_geometry.py`从旧冻结诊断源只改变实验输出/checkpoint位置与源码来源记录；模型输入/计算保持一致，反向替换可精确恢复旧源，见[适配审计](geometry_source_adaptation.json)。它继续使用旧step00生成状态，不用新分支自己的状态冒充训练前后同状态比较。noise probe也只改变checkpoint/output路径，保持输入和Original teacher参照。几何旧收据缺少完整anchor/teacher tensor hash的限制仍在；新旧源码哈希不可简单按相等比较，必须验证上述受控改动。

本目录runtime为冻结快照；306项中仅training入口、resume参数兼容及对应测试3项不同，其余303项原样保留。大权重/原始数据通过既有路径引用，不复制下载。

结果不能只看loss：需同时观察分sigma拟合、同状态action差分、人物/场景/后段重影、MAD/boundary与A/D方向。没有任何自动脚本能仅凭cosine替代画质PASS。C/D局部能力前置仍有效；meeting未替换。


## 2026-10-09 00:20：受控结果比较与CPU报告队列就绪

当前训练10/48，PID1563900/start_ticks255332670仍存活；GPU评测队列PID1615283/start_ticks255366821仍等待完整训练。新增CPU报告队列PID1898518/start_ticks255496357也已核实存活，不占GPU。

[report_density.py](report_density.py)要求完整配对结果：噪声54点逐输入hash及Original完整输出相同；geometry核对固定state/action pair、只读KV、teacher范数，以及诊断入口仅路径/provenance变化的逐字反向核验。旧geometry缺完整anchor/teacher tensor hash的限制继续披露。8项CPU比较器identity/拒绝错误输入检查通过，见[验证收据](report_validation.json)；这些fixture不是新checkpoint效果。

[post48_reports.py](post48_reports.py)将等待8组完整结果，自动输出逐状态CSV/表格、完整39帧对比MP4和全部39帧contact sheet。缺列不填充，错误不自动重试。视频报告明确时延口径、GPU峰值未分解weights/activations、CPU KV与GPU分列；MAD及数值action gate不能自动通过视觉验收。

报告源码在controller启动后冻结；实时状态见[报告队列](post48_reports.json)。截至本记录，尚未生成新候选视频或得到训练后验收结论。


## 2026-10-09 00:40：step16保存与执行协议核验通过

step16 checkpoint于00:37完整写出，[核验收据](checkpoint16_audit.json)通过：208/208 bank模块更新、冻结visual仍零输出、Adam状态finite且step计数为16、动作/chunk/sigma课程及完整逻辑noise RNG重放匹配、306项runtime源hash保持。此前[step3核验](checkpoint03_audit.json)也通过。这些是执行协议检查，不是动作或视觉验收。

训练当前17/48，原PID仍存活并继续。只读checkpoint watcher已正常完成并退出；GPU和CPU报告队列继续等完整48更新，不重启训练。未新增GPU实验或改动架构/训练源。


## 2026-10-09 00:49：完整视频报告器CPU集成检查通过，训练22/48

[检查源码](verify_video_reporting.py)在临时目录中用旧control作为两列identity输入，验证自然GT30/停车场8两类入口。4个临时视频完整解码为39帧、24fps、H264/YUV420P、faststart；14张全部帧图、对应帧图和identity统计匹配，自动视觉PASS保持false。fixture全部删除，未在正式结果中填充候选列；[收据](video_report_validation.json)不代表新模型结果。

原训练与两个队列PID/start ticks再次核实存活，预算/源码保持不变。当前22/48，仍等待真实候选48-update视频与动作验收。


## 2026-10-09 01:08：step32已保存并通过只读核验

[step32收据](checkpoint32_audit.json)验证208/208 bank更新、冻结visual零输出、Adam finite/step32、课程/sigma/noise RNG和306项runtime哈希。训练当前32/48，原训练/两个队列均核实存活；只读观察session5968正常结束，不是训练退出。尚无本分支训练后视频/action结论，不追加预算。


## 2026-10-09 01:45：训练完成，正式评测运行

[最终训练审计](completed_training_audit.json)和[step48 Adam/RNG审计](checkpoint48_audit.json)通过。原训练PID已退出，原controller在GPU1启动自然GT30、GPU0启动54点noise；CPU报告等待完整组。当前快照见[进程记录](live_snapshot.json)，不得仅凭本页推断实时状态。训练比较中mid比旧control低0.94%、high高0.67%，无low且只有chunk2，不能当视频或动作通过。原始训练收据和[统计重算检查](training_report/replay_verification.json)已归档。
