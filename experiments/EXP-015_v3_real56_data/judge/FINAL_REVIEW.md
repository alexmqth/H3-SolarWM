# EXP-015/v1 — Judge最终验收

2026-10-11 08:39 HKT。**ACCEPT WITH LIMITS：四个固定train episode的真实56RGB数据、原始动作与native12+5分块准备通过；0GPU、0模型编码、0训练。** 不宣称已经具备可直接训练的完整V3条件fixture。

## 已核实证据

Judge独立读取原annotations的11keys，全部224行与新数据相同；源视频/标注、确定性选片和所有输出摘要匹配。四MP4共224帧完整解码，832×480、24fps、PTS递增，新I0逐值等于对应第一帧。原17列动作复算完全一致，native spans独立手算通过，首12恰到39RGB、后5到56；大幅改变C2原始动作不改变C1的pool/keys9/script。[独立审核](INDEPENDENT_DATA_AUDIT.json)。

检查了实际构建的select/setpts命令及[Worker源PTS](../SOURCE_PTS_AUDIT.json)：所有56帧索引逐行等于start+5*(j//4)+j%4。s1近邻RGB排序因不同缩放路径存在歧义，不能独立证明错位；原源SHA、正确滤镜、实际选中PTS、旧前39片段及较小重编码差异支持数据时间对齐，不再重建。新I0对旧PNG的MAD为0.031–0.551，不把新旧fixture称为逐字节相等。

Judge查看[四场景首帧、边界及末帧](REAL56_KEYFRAMES.jpg)：真实角色和环境可辨，s3暗部/草遮挡属源内容；没有生成质量结论。四段实际动作：s0 A+S+L、s1 A+S、s2 A+S+J；s3 A40帧→静止8帧→W8帧。Q/E/Space均0，相反key冲突0。F为观测yaw速率派生代理，文字丢失连续量，raw数据保留。**没有真实D视频或反事实GT。**

## 资源与偏差

四段实际产物共8,412,751bytes，磁盘约121GiB，低于1GiB上限/高于60GiB底线，08:50前完成。首次filter错误和后续为限制自动线程而中断的记录保留；最后续建manifest中的15.880秒/2,174,976bytes只是续建进程值，不作为全任务总账。FFmpeg OS线程峰9、续建5超过原≤4线程约定，Worker已披露；后续用4 CPU affinity约束可运行核数，不因该资源偏差重做正确数据。没有GPU或训练调用。

## 后续两个独立闸门

1. VAE前缀：需以同一新56RGB分别编码56与前39，检查前12latent一致性；参考[编码协议说明](ENCODER_PROTOCOL_NOTE.md)。不能用旧39压缩片替代输入。
2. 文本布局：Judge使用实际冻结PackedSequenceBuilder/visible_inputs源码做CPU合成长度复现。同一C1可见prompt不变，仅每个未来action span多1token，裁剪后可见video时间坐标仍整体移动25，head坐标不动。[复现证据](FUTURE_LAYOUT_RISK.json)。这证明直接将变长未来文本先打包再裁剪不能保证位置未来隔离；不否定已有等长A/D冻结证据。尚无真实tokenizer/模型效果结论，禁止暗改prefix/time/位置来绕过。

下一任务EXP-016只做I0/真实video VAE编码、39/56前缀比较和有限roundtrip，先CPU入口后单卡有限GPU授权。动作文本/packed布局、DiT、训练均不执行；完整真实转移训练需在布局协议明确后另立任务。
