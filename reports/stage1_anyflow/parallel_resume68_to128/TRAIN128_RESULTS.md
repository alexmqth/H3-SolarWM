# AnyFlow128训练完成；视频评测中

记录：2026-10-08T14:34:02.044771+08:00。

GPU4–7完成96→128共32次新增optimizer更新。含加载/前后验证1659.72秒；逐更新wall合计1507.65秒。allocated峰值40504.14–40505.49MiB/卡（约39.56GiB）。四replica参数、Adam、logical/CPU/CUDA RNG哈希一致；QKV/out/FFN/refiner全部更新，原visual和target-time保持冻结。

实际保存的预更新step96与来源96的四套adapter、Adam、teacher/update history、固定配置与三种RNG逐项相等，见[恢复审计](resume96_audit.json)。128训练前的8个validation样本记录与96训练后的记录相同。新增更新含512 current forwards、120 physical clean commits、120带梯度历史forward；不含validation或backward checkpoint重算。四卡为同global-batch4的样本并行，不是模型/sequence sharding。

固定0/16/32/64/68/96/128验证的sigma/r/类别/权重逐项一致，见[完整学习曲线](training_curve_00_16_32_64_68_96_128.json)。A endpoint96→128为35.394920→34.422421，D为18.415810→15.185422。原日志未保存actual GPU noise hash；保留这一限制。内部误差下降不能证明视频改善，96已经显示两者可能背离。

控制器已开始128的39f A/D、4/8 steps/chunk评测，完成后无条件停止，不自动增加128以上训练。当前没有完整128视频结果，不报告128的A−D或视觉通过。Stage1仍未验收；旧meeting展示保持。

阶段判断见[Stage1与Stage2边界](../STAGE_BOUNDARY.md)：不要求Stage1预先消除全部长时generated-history漂移；先验证首块和受控历史下的动作/画面，再决定是否针对自生成历史使用Stage2。

## 首条A4评测（完整A/D结果尚未收齐）

A39f、4 steps/chunk的水平光流为−0.751431，方向仍错。全部0–38帧contact sheet和Original/64/128在12/24/30/38帧对应画面已静态查看；约18–20帧起重影，22帧后严重雾化、人物轮廓分解，未见相较64的明确视觉修复。静态全帧复核不是实时播放。该条失败不能代替未完成D4/A8/D8的结果，不报告128的A−D。源视频保留完整39帧，评测JSON和两张复核图归档同目录。
