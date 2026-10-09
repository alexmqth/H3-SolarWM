# 相同48更新，采样density对动作差分的影响

固定状态来源：gt。18点当前chunk A/D反事实；各模型独立构建历史KV，A/D内部只读。

| chunk | whole cos shift12 / 2.22 | A/D delta cos shift12 / 2.22 | delta norm ratio shift12 / 2.22 |
|---|---:|---:|---:|
| all | 0.989033 / 0.989268 | 0.017641 / 0.008237 | 0.653948 / 0.631022 |
| 0 | 0.989598 / 0.990021 | 0.030018 / 0.029546 | 0.851171 / 0.849807 |
| 1 | 0.988739 / 0.988898 | 0.011285 / -0.013008 | 0.594269 / 0.530855 |
| 2 | 0.988762 / 0.988886 | 0.011621 / 0.008172 | 0.516405 / 0.512403 |

模型计算源相同，诊断入口只作经逐字反向核验的输出/checkpoint路径与来源记录变动。逐state/action/endpoint哈希和teacher范数一致；旧收据没有完整anchor或teacher输出hash，不能把标量核验称为完整tensor逐bit证明。
Original仍双向重算历史，student使用已commit的causal KV；插值状态不是实际solver中间状态。不能由cosine单独判定视频方向或画质。

[逐状态CSV](metrics.csv) · [完整记录](metrics.json)
