# C准备：已有AnyFlow128的自适应权重及输出梯度核查

CPU-only只读核查，不启动训练或新的GPU工作。使用旧AnyFlow128的512个实际样本，不是当前真实ABot FM48，也没有宣称可靠causal checkpoint上的C已经完成。

## 直接结果

512个样本的adaptive scale与冻结训练函数逐值重放误差为0；weighted loss最大相对重放误差0。

| 类型 | noise | n | Raw loss均值 | Adaptive scale中位数 | 输出梯度L2均值 | 去掉adaptive时的L2均值 | 两列L2之和比 |
|---|---|---:|---:|---:|---:|---:|---:|
| diffusion | all | 256 | 0.0891442 | 1 | 0.000415 | 0.000415 | 1 |
| diffusion | low | 6 | 0.431764 | 1 | 0.000873224 | 0.000873224 | 1 |
| diffusion | mid | 39 | 0.0954468 | 1 | 0.00116241 | 0.00116241 | 1 |
| diffusion | high | 211 | 0.0782366 | 1 | 0.000263823 | 0.000263823 | 1 |
| endpoint | all | 128 | 54.4889 | 0.00473842 | 8.58094e-05 | 0.00376126 | 0.022814 |
| endpoint | low | 2 | 0.507591 | 0.16559 | 0.000131084 | 0.000798262 | 0.164212 |
| endpoint | mid | 18 | 1.134 | 0.102082 | 0.000381735 | 0.00378454 | 0.100867 |
| endpoint | high | 108 | 64.3811 | 0.00326492 | 3.565e-05 | 0.00381226 | 0.00935143 |
| flow_map | all | 128 | 1.67849 | 0.433245 | 0.000295045 | 0.00084392 | 0.349612 |
| flow_map | low | 3 | 0.364125 | 0.197536 | 0.000218115 | 0.00103384 | 0.210977 |
| flow_map | mid | 12 | 0.211703 | 0.444835 | 0.00102981 | 0.00159928 | 0.643921 |
| flow_map | high | 113 | 1.86915 | 0.455667 | 0.00021906 | 0.000758663 | 0.288744 |

low≤0.240781，mid≤0.689441，high>0.689441。跨训练状态的描述性分组，不是固定输入验证或独立样本统计。

## 计算的是什么

实际loss为`alpha * w * mean(e²) / B`，其中B=4；e中finite-difference target和alpha均detach。因此对本次有梯度的预测velocity，`||d loss/d prediction||₂ = 2*alpha*w/B*sqrt(raw_loss/N)`。N由实际历史teacher latent shape与当前chunk长度得到，包含末chunk长度2，非统一假设为5。

本表仅重建输出端梯度范数。真实参数梯度还要乘H3的Jacobian；不同样本可能方向冲突，之后还有clip/Adam。列中范数相加不等于合并后的梯度范数，不能把百分比叫做参数梯度贡献或训练预算占比。去掉adaptive的一列仅是同一保存状态下的数学反事实，没有运行无adaptive训练。

3个实际标量/维度的autograd核查确认输出导数公式；故意不detach的负对照会额外乘`eps/(raw+eps)`，冻结源码实际没有此错误。这不是实际33B参数梯度实验。

## 对后续的意义

高raw residual的非对角样本会被自适应缩放压低输出梯度，weighted loss接近FM参考不能证明finite-map误差已小。这个行为符合现有实现，尚不能判断是否过度压制，更不证明它导致普通FM的action geometry失配。

此处raw residual包含有限差分导数项，不是单纯velocity MSE或GT endpoint误差。高噪声endpoint的108样本raw均值64.381、weighted均值0.06978；同状态下输出梯度范数之和约为数学上去掉adaptive的0.935%。这既提示不能只看weighted loss，也说明直接移除adaptive可能放大高残差梯度，不能据此自动改训练。

进入可信causal checkpoint的C验证时，继续同时记录raw residual、diagonal保真、teacher/GT端点距离与composition；若要调整adaptive策略，必须单变量比较真实参数梯度和完整视频，不能直接删除权重。B局部action前置仍未通过，本核查不授权自动扩训128/136。

初版CPU负对照复用了已释放的计算图而失败；改为从同一prediction叶子独立重建raw loss后通过。首次源码/失败记录保留在initial_audit.py及initial_failure.json；没有改生产模型或运行中训练。

[完整收据](audit.json) · [512样本](samples.csv) · [历史训练记录](training128.json) · [检查源码](audit.py)
