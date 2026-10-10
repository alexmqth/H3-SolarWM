# EXP-013/v1 Judge 最终验收

2026-10-11 07:25 HKT。**ACCEPT：CPU 数据审计、四张确定性训练初图和独立后续设计完成。** 本任务 GPU/训练/下载均为 0；不构成新的 AnyFlow 能力证据。

## 独立核查与结论

[独立审计](INDEPENDENT_DATA_AUDIT.json)覆盖全部24 clip的来源摘要、936帧完整解码、首PNG与视频首帧逐像素相等、936行×11二值按键与原annotations直接比对、源帧索引和四张确定性选图。连续6列由Worker从原helper重新计算，全24数组误差为0；不把这部分写成Judge独立实现的连续动作验证。

实际6 episodes：4 train/16 clips、2 validation/8 clips。episode和源视频摘要不跨split，Worker同时核对annotations摘要隔离。工业与村落两个validation episode此前已观察，今后仅称固定回归集。停车场与这些episode的原源关系未知，不声称统计独立。

最终inventory.jsonl SHA `b9082c499e09b3006b593b0d55821dd5ed8e2e65109a6399379d6c8508c85896`、candidate SHA `b6af2b0208a5dcf83c02f76fa826e5a03ba52fa9bd83de357f9474b5efb257b6`与先前独立审核一致。新增inventory.json与JSONL逐条内容相等，SHA `887532a37144a06cc7eeabb8efd88bdd560ee05815b192b044b60c885595c3b2`，无需重新解码相同视频。

三条邻帧像素MAD歧义已由原FFmpeg生产流程重建解决：重建MP4与已存文件逐字节摘要相等。无需继续追查缩放/压缩导致的像素代理差异。

## 对未来训练有实际影响的限制

- 四张初图按每train episode最早target=A选定，不按生成结果换图。真实按键为A+S+L、A+S、A+S+J、A。未来纯A/D教师条件是合成反事实，不是录屏GT。
- 旧编码为Dual Anchor，不能作为native Single I0输入。24片只有39RGB/C1，没有现成C2 GT。
- 未来训练使用冻结teacher的C1 latent，由当前student重建自己的clean KV；这与student自己采样C1的闭环评估不同。
- 独立student从V3初始化；不继续挽救AF2 step32。32update和训练预算仍是草案，没有GPU训练授权。
- 评估优先复用EXP-006/011已有FM8视频，协议匹配时新AF预算126forward/15decode，不重算FM8。当前AF4/DMD失败配置保持归档。

## 下一步与ROI判断

发布独立EXP-014：先准备四train图native输入，再生成冻结FM30的C1与AA/AD C2教师目标。先验收第一图，再决定其余三图，预算和GPU marker另列。这个任务回答教师是否能提供可信训练目标；普通画质缺陷按PARTIAL记录，持续结构/动作失败则停，不通过筛图和调参制造可用数据。此刻尚无新GPU调用。

完整交付见[Worker报告](../WORKER_REPORT.md)、[设计](../FUTURE_GPU_DESIGN.md)和[候选联系表](../candidate_contact_sheet.jpg)。
