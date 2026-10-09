# 相同48更新，采样density对动作差分的影响

固定状态来源：step00_generated。18点当前chunk A/D反事实；各模型独立构建历史KV，A/D内部只读。

| chunk | whole cos shift12 / 2.22 | A/D delta cos shift12 / 2.22 | delta norm ratio shift12 / 2.22 |
|---|---:|---:|---:|
| all | 0.995210 / 0.995354 | 0.009336 / 0.013624 | 0.673216 / 0.693159 |
| 0 | 0.992853 / 0.993307 | -0.030470 / -0.029165 | 0.729056 / 0.733966 |
| 1 | 0.996163 / 0.996237 | 0.024807 / 0.032031 | 0.676252 / 0.684013 |
| 2 | 0.996615 / 0.996517 | 0.033670 / 0.038007 | 0.614341 / 0.661498 |

模型计算源相同，诊断入口只作经逐字反向核验的输出/checkpoint路径与来源记录变动。逐state/action/endpoint哈希和teacher范数一致；旧收据没有完整anchor或teacher输出hash，不能把标量核验称为完整tensor逐bit证明。
Original仍双向重算历史，student使用已commit的causal KV；插值状态不是实际solver中间状态。不能由cosine单独判定视频方向或画质。

[逐状态CSV](metrics.csv) · [完整记录](metrics.json)
