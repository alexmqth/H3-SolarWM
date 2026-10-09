# FM48：固定GT状态的分噪声velocity/endpoint误差

2个held-out自然场景×3个chunk×9个sigma，54个逐一匹配状态。8个点来自实际8-step shift2.22网格，另补30-step末端sigma0.071108。不是新增训练、A/D反事实或视频质量评测。

所有输入/noise/history/anchor/prompt/source hash一致；两独立GPU上Original teacher完整输出hash逐点一致。每个checkpoint用自己权重构造GT clean history KV，预测不写cache。Original依然双向重算历史，所有分支统一SDPA和匹配条件。

| sigma | Original raw MSE | causal0 raw MSE | causal48 raw MSE | Original endpoint MSE | causal0 endpoint MSE | causal48 endpoint MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000000 | 0.163484 | 0.276204 | 0.277907 | 0.163484 | 0.276204 | 0.277907 |
| 0.939540 | 0.099822 | 0.131455 | 0.124893 | 0.088117 | 0.116040 | 0.110247 |
| 0.869452 | 0.084698 | 0.106819 | 0.103374 | 0.064027 | 0.080749 | 0.078145 |
| 0.787234 | 0.085424 | 0.107922 | 0.105091 | 0.052941 | 0.066883 | 0.065129 |
| 0.689441 | 0.094714 | 0.118398 | 0.115804 | 0.045020 | 0.056278 | 0.055045 |
| 0.571184 | 0.115900 | 0.140939 | 0.138355 | 0.037812 | 0.045981 | 0.045139 |
| 0.425287 | 0.157747 | 0.186783 | 0.183977 | 0.028532 | 0.033783 | 0.033276 |
| 0.240781 | 0.244133 | 0.281336 | 0.277918 | 0.014154 | 0.016311 | 0.016112 |
| 0.071108 | 0.514912 | 0.575279 | 0.569519 | 0.002604 | 0.002909 | 0.002880 |

每个sigma均是6状态等权平均；不是按训练分布或推理轨迹出现概率加权。

Velocity目标为真实GT的noise−clean。Endpoint是单次预测构造的 z_t−sigma*v，不是完整8/30步生成结果；其MSE在同一插值状态上等于sigma²×velocity MSE。两列帮助区分误差尺度，不是两个独立验证集。GT conditional flow本身可多模态，Original对某个单样本GT的误差也不为0，必须与Original对照，不能将所有低sigma误差归因于causal训练。

| 训练noise段 | 样本数 | 样本比例 | 权重总量比例 | 加权loss比例 |
|---|---:|---:|---:|---:|
| low | 4 | 2.08% | 2.06% | 6.51% |
| mid | 35 | 18.23% | 42.88% | 51.81% |
| high | 153 | 79.69% | 55.05% | 41.67% |

这些训练loss来自不同更新、数据、噪声，不能画成固定输入学习曲线；加权loss比例不等于参数梯度比例。Shift同时影响sigma采样density和Gaussian normalization，表中单独保存shift12/2.22的固定sigma weight以供后续受控设计。当前未改sampling/weight。

[逐状态CSV](metrics.csv) · [完整统计](analysis.json) · [训练覆盖](training_coverage.json)
