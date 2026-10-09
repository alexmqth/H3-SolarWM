# 实验报告：工程可行，动作与质量联合验收尚未通过

[按模型／配置分三类的展示](../meeting/model_types/README.md)：零更新routing、普通causal adaptation与AnyFlow分别提供对比视频；history来源和KV协议另列，不把类型等同于验收通过。

[后续训练视频与数据来源总览](../meeting/DATASET_AND_ANYFLOW.md)：补齐真实数据/FM 与 AnyFlow 历史单条输出；实验结论不变。

**最终结论（2026-10-09）：** 真实H3上的causal chunk、persistent KV与长rollout执行已跑通；未证明四向动作完整保真、20秒稳定生成或端到端加速。最后一轮局部E2两臂各4更新已全部评完，新增动作排序损失没有一致优于FM-only，按负结果冻结。

## 1. 实验与演示版本分开

| 证据 | 协议 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 会议124f主片 | 2026-10-06 RGB visual checkpoint；persistent CPU KV；generated history；8steps/chunk | 真实因果执行；124f相对旧版的结构改善 | 不是AnyFlow/E2；A/D gate失败；没有速度提升 |
| 同checkpoint243/481f | W、同首帧/prompt/seed/noise，真实长rollout | 长时漂移诊断；20秒明确失败 | 不能叫长视频稳定性成功 |
| Stage1/AnyFlow系列 | 显式目标时间与有限区间映射，含真实33B训练/136对照 | AnyFlow实现及诊断链路可运行 | 内部一致性不能替代动作/画质验收 |
| 旧Stage2-lite | self-rollout、trainable fake-score、DMD surrogate，共享底座 | 精简角色闭环可执行 | teacher仍为causal cached、endpoint replay近似；非完整SolarWM Stage2 |
| 新DMD/FMBS工程准备 | tiny真实H3类的梯度、符号与角色隔离测试 | 支撑后续正确实现 | 没有新的33B完整DMD效果结论 |
| 最新E1/E2 | T2/N局部窗口30step，reference/GT history，无persistent hidden KV | 检查动作信息流与观察后果监督 | 非124f自由rollout，不能继承主片性能或缓存结论 |

主片checkpoint、哈希与指标来源见[DEMO_PROVENANCE](../meeting/DEMO_PROVENANCE.md)。历史实验原始报告不重写；本页是最终状态入口。

## 2. 124帧主Demo及历史测量

[W并排主片](../meeting/annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4) · [W/S/A/D grid](../meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4) · [完整指标](../meeting/METRICS.md)

左右同首帧、prompt、action序列、seed13、初始video/audio noise、832×480、124RGB、24fps。原始30整段steps；causal为8steps/chunk×8chunks=64noisy forwards＋8clean commits。当前仅有历史单次记录，**没有统一独占硬件下warmup重复均值**。

| 方法 | 端到端范围 s | 峰值allocated MiB | CPU raw KV MiB | 首内部chunk s | A flow | D flow | A−D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original H3 30-step | 441.5–454.2 | 39925 | 无跨chunk持久KV | 不适用 | +1.077 | −1.602 | 2.679 |
| RGB-main causal | 673.4–767.9 | 31570–39940 | 13509 | 41.8–57.4 | −0.784 | −1.007 | 0.223 |

GPU峰值未拆分权重/激活/临时KV，不能相减推算。首chunk是内部生成/提交计时，不是端到端首帧可播放延迟。RGB anchor和CPU传输有开销，forward总数及active sequence都不同，不按30/8推导速度。

Frame MAD和boundary MAD是描述性活动/连续性指标；低MAD可能来自模糊或冻结。Farneback为图像水平运动代理，不是严格动作准确率。没有为此主片跑FVD、LPIPS/PSNR或VBench，不补造质量分数。历史原始JSON和公平性说明见[FAIRNESS](../meeting/FAIRNESS.md)。

## 3. 必须保留的失败证据

- [动作较强旧版 vs 视觉较完整新版](../meeting/action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)：旧fixed-mix A−D约0.453且符号正确，但明显漂移；RGB-main A−D约0.223且A符号错误。多项配置同时改变，不是单变量ablation。
- [20秒完整Original vs causal](../meeting/long_horizon/original_vs_rgb_visual_W_20s_481f.mp4)：同一RGB checkpoint，约10秒严重雾化，15秒后人物/场景难辨。完整481帧保留，不能把执行完成解释为稳定。
- [最新E2 D-history/current A](../reports/stage1_anyflow/01_real_video/real_transition_windows/review_step4/parking_historyD_currentA_comparison.mp4)：局部A方向保留，但手臂/躯干残影仍在。此处左列是Original权重的局部N，不是Original完整双向生成。

## 4. 最终E2受控结果：冻结在4更新

[完整结果与全部六条三列视频](../reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md)。两臂都从Original H3＋released action LoRA出发，只训练tail8 QKV/out的10,092,544参数，LR2e-5、logical batch2，初始化一致；唯一实验变量为普通FM或FM＋观察动作后果排序。排序错误动作没有对应反事实GT，不能解释成完整反事实监督。

T2/N每sigma重新计算可见hidden；12latent当前窗、single I0、30steps、shift2.22。停车场历史来自Original生成；两条真实GT-history保留联合动作，当前速度派生camera F属于oracle后果条件，不证明实时输入可用。

| 固定history | 参数bank | A flow | D flow | A−D |
|---|---|---:|---:|---:|
| A-history | Original local N | +1.645495 | −1.546733 | 3.192228 |
| A-history | FM-only4 | +1.657549 | −1.395918 | 3.053467 |
| A-history | FM+action4 | +1.678612 | −1.376481 | 3.055094 |
| D-history | Original local N | +0.614104 | −0.673822 | 1.287926 |
| D-history | FM-only4 | +0.699057 | −0.667578 | 1.366635 |
| D-history | FM+action4 | +0.657532 | −0.616613 | 1.274146 |

两份history中方向符号均保留，但动作项没有一致增益；两组当前A仍有重影，当前D结构相对完整。两条GT局部例子中，一组首边界灰度MAD由24.16降至20.89/20.85，另一组约24.05→24.18/24.16；人物姿态/尺度突跳仍在。完整738当前RGB的静态全帧、原尺寸人物和边界检查已完成，不声称实时播放评审。

Held-out正确动作FM：0.25976668→0.25969061/0.25969900，改善不足0.03%；正确动作优于交换动作的排序由5/8变为4/8、4/8。两个状态×四sigma不是八段独立视频，不在验证集上调整margin/lambda。

两臂训练wall284.76/416.62s，峰值allocated均27.91GiB。视频评测共360采样＋12identity forwards，额外48次held-out诊断。Original→step0误差和梯度replay误差均为0，冻结runtime307文件核验通过。每组两条评测的耗时、内存和完整收据见原报告；这些局部作业成本不能和124f主片混表。

**决策：No-Go。** 不扩到16更新，不启动AnyFlow/Stage2；也不从4更新推断动作监督不可能有效。覆盖审计发现8个训练microbatch无纯A/D、3个含相机控制；后续研究应先审查监督覆盖、时序和可靠同状态后果。当前任务只收尾提交，不开始该新实验。

## 5. 最终可复现性验收

步骤和依赖入口见[REPRODUCE.md](../REPRODUCE.md)，本次独立环境验收记录见[final_acceptance](../reports/final_acceptance/README.md)。验收分开记录干净源码/依赖、真实权重推理、KV测试与小H3训练smoke；通过这些工程检查不改变上述画质No-Go结论。

## 6. 技术判断

应继续研究局部双向、跨窗因果的信息流与action-dependent prefix重算规则，并用可靠动作后果验证模型是否学会正确变化。路线依次是可信causal30step、AnyFlow4/8step、on-policy Stage2。局部能力尚不可信时追加DMD预算，不能清楚区分动作表征问题和长时generated-history分布问题。
