# EXP-005：V3 Strict Causal KV 的 Sliding Window 验证

**模型：** Original H3 + released action LoRA；没有新增训练、AnyFlow或DMD。本实验把V3 Baseline从124帧推进到真实窗口淘汰后的141/158帧，分别评估 Global RoPE (SW-G) 与读取时Local RoPE (SW-L)。原Baseline和全部旧结果保持冻结。

## v1 CPU协议阶段（已由Judge验收）

先实现显式`12→5` latent分块、最近5祖先缓存审计，以及SW-G/SW-L独立入口。14项CPU测试通过；旧区间index2–5的attention数值和模型实参匹配冻结EXP-003入口，首淘汰后的结构性小张量测试通过。这只证实已测协议，不代表33B画面/动作质量。历史原版文档逐字节保存在[previous_stage1](stage2/previous_stage1/)；[v1 Judge结果](judge/FINAL_REVIEW.md)。

## v2有限GPU阶段

用户批准后按[冻结任务书v2](taskbook_v2.md)分G0→G1→L1执行；每一段独立锁预算和输入/源码SHA。30步native FM、Single I0、own-action与current-prefix feedback、sigma0 clean commit不改，全部使用真实自生成历史。47-latent长输入是[经审计的显式外推](stage2/long_fixture_review.json)：旧37latents及其prompt/position/noise不变，新增动作embedding与原生时间网格明确延伸；**它不等于H3默认full-length builder重打包**。

| 阶段 | 执行/验收 | 核心发现 | 限制 |
| --- | --- | --- | --- |
| [G0](stage2/G0_RESULT.md) | 已执行，Judge PASS | 旧/新同状态velocity和C6 endpoint完全相等；124RGB逐像素相等 | 淘汰前回归，不是新能力 |
| [G1 SW-G](stage2/G1_RESULT.md) | 已执行，Judge有限可行性PASS、质量PARTIAL | 真正淘汰C1后A/D C7方向相反；两路均续到158帧 | 块边界跳变、A短暂重影、D-C8持续半透明残影；单场景seed |
| [L1 SW-L](stage2/L1_RESULT.md) | Judge接受工程/执行证据，画质PARTIAL；不升级 | 同状态Local位置使velocity relative RMS改变0.1825；A/D方向仍可辨 | 四段帧内MAD约9–10，高于G1的约4.6–4.9；透视、亮度和残影问题，无视觉收益证据 |

G1同一C6历史下C7 A/D的水平光流为`+0.819 / −1.575 px/帧`；C8接各自历史后为`+0.768 / −1.185`。L1同历史C7为`+1.120 / −0.598`，方向还在但分离度下降。flow只是运动代理，不能替代完整视频评估。G1使用123forward、4VAE、1068.666s、峰值allocated26.121GiB；L1为124forward、4VAE、1007.969s、峰值26.621GiB。两阶段工作量不同，不能当公平E2E速度对照。只有**历史video raw KV**被W5约束为14,164,800,000 bytes，完整prefix/RGB/latent与VAE成本不因此有界。

观看：[G1 A继续 vs D切换，158帧并排](artifacts/stage2/G1/G1_A_vs_D_158.mp4) · [G1-vs-L1 A](artifacts/stage2/L1/G1_vs_L1_A_158.mp4) · [G1-vs-L1 D](artifacts/stage2/L1/G1_vs_L1_D_158.mp4)。原始大型CPU KV与latent保持在`H3-World/outputs/EXP-005_v3_sliding_window/`，不复制到提交目录；精选视频及小型日志归档于`artifacts/stage2/`供GitHub查看，逐文件hash见`artifact_manifest_stage2.json`。

## 文件与复现入口

- [stage2目录](stage2/)：G0/G1/L1独立runner、冻结配置、授权、source manifests、实际结果与Judge审核。授权文件绑定runner/config/manifest SHA；任何阶段失败均计入预算，不自动重试。
- [协议](PROTOCOL.md)、[manifest](MANIFEST.md)、[机器指标](metrics.json)、[Worker完整报告](worker_report_v2.md)。
- 冻结V3参考：[EXP-002](../EXP-002_native_cached/README.md)、[EXP-003](../EXP-003_native_cached_124/README.md)。

当前所有生成结论只覆盖停车场seed13及特定A持续/切D路径。工程正确、动作响应、视觉稳定与效率分别报告；G1可行性成立不意味着成熟长视频或总体E2E加速。L1的Local重映射仍冻结历史hidden states，不等价于历史重算；现有视频没有证明其改善了G1。Judge已停止本轮Local调参/C9，后续需独立任务授权。

[Judge阶段二最终审核](judge/STAGE2_FINAL_REVIEW.md) · [完整协议与风险](judge/STAGE2_PROTOCOL.md) · [总资源账本](judge/stage2_summary.json)
