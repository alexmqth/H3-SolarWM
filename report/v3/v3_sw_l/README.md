# V3-SW-L — Sliding Window + Local RoPE

**EXP-005/v2已完成：工程/cache PASS，生成质量PARTIAL；当前无训练Local方向归档，不替代SW-G。** 正式V3 Baseline保持冻结。

保留SW-G的strict causal、persistent raw KV、最近5祖先、30步及全部原生条件，只改变video位置。C7/C8最老latent起点b=12/17；位置从Global `O+tau(j)`改为`O+tau(j-b)`，prefix保持Global，空间坐标不变。canonical Global metadata不表示历史隐藏状态来自Global；不等价于重算历史。此H3路径没有SolarWM camera PRoPE。

## 实际结果

- 同Global历史/动作/噪声开始，两条路径各到158帧；历史RGB不变，真实淘汰/全部50层精确indices/14,164,800,000 bytes KV检查通过。
- A C7/C8 flow +1.120/+0.343，D −0.598/−1.269，动作方向信号仍在。
- 人物可辨，但四段新增帧均有明显场景重排、亮度跳变或残影。相比Global没有动作/画质联合收益；不追加C9或调参。
- 同G1-A C7历史、同C8 noisy state的Global/Local velocity relative RMS差异18.25%，历史不变。位置修改确实影响计算，不能称无损。
- 单停车场seed13、158帧约6.58秒、零新训练。有限负结果不代表经过专门训练的Local路线永远无效。

[Judge最终审核](../../../experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md) · [完整协议](../../../experiments/EXP-005_v3_sliding_window/judge/STAGE2_PROTOCOL.md) · [A的Global/Local对比](../../../experiments/EXP-005_v3_sliding_window/artifacts/stage2/L1/G1_vs_L1_A_158.mp4) · [D对比](../../../experiments/EXP-005_v3_sliding_window/artifacts/stage2/L1/G1_vs_L1_D_158.mp4) · [V3总览](../README.md)
