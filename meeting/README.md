# 会议展示包：主片、失败证据与5分钟技术判断

**按模型类型展示：** [Original＋causal routing／causal adaptation／AnyFlow](model_types/README.md)。四条完整短片按实际训练目标分类；单窗口、固定history、自由rollout另作评测条件。

[只看AnyFlow/DMD之前：因果化版本、完整视频及联合验收状态](../docs/CAUSAL_BASELINE.md)。单窗口正控、局部参考history候选和124帧缓存版本分别固定，不能互相替代。

**后续训练展示：** [数据集视频 / AnyFlow 结果与全视频索引](DATASET_AND_ANYFLOW.md)。按实际数据来源、objective 和 history 条件区分；包含本次补齐的单条 MP4。

新增：[昨晚真实视频训练与机制进展展示](OVERNIGHT_PROGRESS.md)，集中列出两轮48更新、GT/generated-history完整对照与最新E2，明确报告/视频已同步、新训练权重仍在本机。

**当前结论：causal/KV工程可运行；没有证明动作完整保真、20秒视觉稳定或整体加速。** 最新E2已全部评完，普通FM与新增动作损失各4更新，动作项无一致收益，按负结果冻结。

会议主视频保持10月6日的`visual_online_rgb_tail16_endpoint_ad2`，**没有**替换成最新E2或AnyFlow。包内对应`checkpoints/visual_rgb_tail16/`的两份adapter；[DEMO_PROVENANCE.md](DEMO_PROVENANCE.md)和[JSON哈希清单](DEMO_PROVENANCE.json)给出来源、完整配置与验证。

## 现场播放顺序

1. [Original vs causal W，124帧主片](annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4)：左30整段steps；右8steps/chunk×8chunks，64noisy forwards＋8clean commits。约5秒，先播再解释。
2. [同checkpoint的20秒完整失败片](long_horizon/original_vs_rgb_visual_W_20s_481f.mp4)：主动保留后段，展示10秒后的严重漂移。视频能播放不等于视觉合格。
3. [最新E2固定D-history/current A，三列诊断](../reports/stage1_anyflow/01_real_video/real_transition_windows/review_step4/parking_historyD_currentA_comparison.mp4)：Original权重局部N / FM-only4 / FM+action4；动作响应在，人物重影仍在。它是42当前帧局部实验，不是124帧自由生成。

备选：[四方向主grid](annotated/h3world_rgb_stable_action_grid_124_timed.mp4)、[旧动作较强版与视觉版的取舍](action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)、[10秒对照](long_horizon/original_vs_rgb_visual_W_10s_243f.mp4)。5分钟内不必全播。

## 配套材料

- [MEETING_SCRIPT.md](MEETING_SCRIPT.md)：含视频时间的5分钟讲稿，末尾有追问备答。
- [SLIDES.md](SLIDES.md)：6页展示提纲。
- [METRICS.md](METRICS.md) / [CSV](METRICS.csv)：**旧RGB主片**124f单次指标；没有warmup重复均值，不归因于E2。
- [FAIRNESS.md](FAIRNESS.md)：输入公平性、步数、计时和指标限制。
- [最新E2完整报告](../reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md)：六条局部对照、held-out拟合、视觉评审。
- [最终可复现性验收](../reports/final_acceptance/README.md)：新环境、源码补丁、KV与训练smoke、真实推理收据。

旧主片使用fixed-mix action adapter和latent dual，没有RGB visual adapter，保留在[legacy diagnostics](diagnostics/legacy_fixed_mix/README.md)。新旧协议同时改变adapter、anchor、routing，不能把视觉差异只归因于某个模块。RGB-main只是124帧相对改善，A/D gate和20秒稳定性都失败。

主片原始JSON见[source_metrics/rgb_visual](source_metrics/rgb_visual/)；20秒测量与观察见[long_horizon](long_horizon/README.md)。这些历史测量不被最新E2数字替代。
