# 三个研究分支：已做什么，失败说明什么

| 分支 | 完成的工作 | 状态与结论 |
|---|---|---|
| A Causal Mechanism Diagnostics | 5/7/12分块，clean/N，I0/anchor，action/public prefix，KV/time/RoPE/VAE | 实现与固定状态诊断通过；Original动作传播在strict图中改变，非单纯cache错误。默认数值路径和受控零误差路径明确分开 |
| B Causal Adaptation & Action Recovery | tail QKV、fixed-mix、online replay、action-delta、RGB+endpoint、真实ABot FM48/E2 | 确实训练过；有视觉改善或局部动作变化，但没有一致动作+视觉+长时通过 |
| C AnyFlow & DMD Explorations | TF-AnyFlow有限区间与r=t，16/64/128/136训练、4/8步；fake-score critic/DMD-lite；FMBS/梯度准备 | implementation verified / preliminary trained；旧方案quality failed；完整V3 AnyFlow/Stage2 not yet validated |

## 可现场播放的证据

- [fixed-mix动作较强、视觉较差；RGB反向取舍](videos/B_fixed_mix_tradeoff_124.mp4)：Original / fixed-mix / RGB三列，124f，不是V1零训练主片。
- [真实ABot FM48](videos/B_real_ABot_FM48_39.mp4)：Original / FM0 / FM48，停车场评测；FM48训练数据是真实ABot，而所展示场景是停车场外部正控。
- [AnyFlow16 vs matched FM16，4步失败](videos/C_anyflow16_4step_failure.mp4)：39f完整后段保留。伪标签来自两条Original A/D，不是ABot真实AnyFlow训练。
- [旧Stage2-lite critic/DMD失败](videos/C_dmd_lite_failure.mp4)：4 updates的小预算探索，人物分解；后续RGB集成改善结构但没有恢复动作。不能称完整SolarWM Stage2。

64/128/136均有实际结果和训练记录；增加预算没有把4步质量与A/D联合门槛做过。真实ABot目录虽放在stage1_anyflow历史根下，已核对主要objective是FM或FM+action，不凭目录名升级为AnyFlow。

完整证据在仓库`branches/`及历史`reports/`，汇报包仅保留上面的代表片。[首页](../README.md) · [路线](../roadmap.md)
