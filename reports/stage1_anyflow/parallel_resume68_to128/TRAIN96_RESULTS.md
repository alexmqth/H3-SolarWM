# shift12 AnyFlow96：训练完成，A/D视频评测进行中

记录：2026-10-08T13:56:54.262012+08:00。这份报告是训练结果，不是画质验收。

GPU4–7完成68→96共28次新增更新，加载及前后验证共1491.87秒；逐更新wall合计1327.22秒。Allocated峰值约39.55GiB/卡，四replica参数/Adam/logical-CPU-CUDA RNG哈希完全一致。实际step68预更新恢复全部检查通过。

## 相同固定验证点的raw endpoint loss

| optimizer updates | A | D |
|---:|---:|---:|
| 0 | 59.624653 | 22.804770 |
| 16 | 54.556305 | 24.329163 |
| 32 | 59.536766 | 22.905317 |
| 64 | 61.048344 | 19.990650 |
| 68 | 58.724133 | 20.841051 |
| 96 | 35.394920 | 18.415810 |

96次A endpoint从64次61.048降到35.395，D19.991→18.416；diffusion与general-map raw也略降。所有点的实际sigma/r/样本类型/权重匹配，validation generator固定1001；原训练未记录actual GPU noise hash，不把计划噪声哈希当实际证据。weighted total会自适应缩放，不能拿它代替视频和动作效果。

## 训练执行的实际计数

这28次更新记录的current forward合计448，物理clean commits 120，带梯度历史重建forward 120。不含validation或反向checkpoint重算；四卡各有模型/历史副本，不是SP。全局logical batch仍4、原AnyFlow每样本4次forward未简化。

已核对旧visual冻结、target-time可训练参数0且不变，43,237,376参数bank的QKV/out/FFN/refiner四组都更新。恢复过程中只改变physical GPU和继续总步数，没有更换数据、LR、loss或anchor/采样协议。

已开始source step96的39f A/D、8 steps/chunk真实generated-history评测。先收齐完整视频、同帧与方向；若A>0/D<0/A−D>1则补4步并停待视觉/匹配FM/新seed/动作切换，否则按既定预算继续最多128。训练完成与内部残差下降都不能自动算Stage1通过。

[训练记录](training96.json) · [完整固定验证曲线](training_curve_00_16_32_64_68_96.json) · [真实回载审计](resume68_audit.json)

## 首条A/8-step视频完成

水平光流−0.483647，A仍方向错误。已查看全部39帧和Original/64/96在12/24/30/38帧的静态对应画面：人物与车库保留，约20帧后人物模糊、重影和透明感仍在；相对64次没有明确视觉修复。静态复核不是实时播放评审。D8尚在生成，不提前报告完整分离度。
