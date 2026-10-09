# 展示视频与 checkpoint 对应表

本次收尾**保留原会议主视频**；它不是最新 E2 的效果。小型 adapter 和首帧的 SHA256、主视频及失败视频哈希见 [DEMO_PROVENANCE.json](DEMO_PROVENANCE.json)。提交包两份主 adapter 已与原实验源文件逐字节哈希核对。

| 项目 | 会议主视频 / 20秒失败片 | 最新 E2 三列局部视频 |
|---|---|---|
| 版本 | `visual_online_rgb_tail16_endpoint_ad2`，2026-10-06 | `real_transition_windows`，2026-10-09 |
| checkpoint | `checkpoints/visual_rgb_tail16/{causal_adapter.pt,action_adapter.pt}` | Original local N / FM-only step4 / FM+action step4 |
| 训练 | tail16 visual QKV 在线 teacher replay 2更新，fixed-mix action residual冻结 | released action LoRA tail8 QKV/out，两臂各4更新 |
| 训练目标 | 普通 FM/replay＋endpoint辅助；**没有AnyFlow或DMD训练** | 普通FM / FM＋观察动作后果排序 |
| 推理 | generated history，5 latent/chunk，8 steps/chunk，RGB dual anchor | reference/GT history，12 latent局部窗，30 steps，single I0，N历史加噪 |
| KV | persistent raw KV on CPU，clean commit | 每sigma重算可见hidden，**不复用persistent hidden KV** |
| 时长 | 主视频124帧；同checkpoint失败片481帧 | 停车场42当前帧、GT39当前帧 |
| 可得结论 | 124f相对旧版结构改善；A/D gate失败；20s严重漂移 | A/D方向保留，当前A仍重影；动作损失没有一致额外收益 |

原checkpoint目录：`H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/`。历史checkpoint内的metadata `status=running`是当时保存时的标签，不表示当前还有训练运行；实际主视频完成记录见 [W setup](source_metrics/rgb_visual/causal_W_setup.json) 和 [W run](source_metrics/rgb_visual/causal_W_run.json)。

主片：

- [Original vs causal W，124帧](annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4)。左Original30整段steps，右8steps/chunk×8chunks＋8clean commits。
- [W/S/A/D总览](annotated/h3world_rgb_stable_action_grid_124_timed.mp4)。同首帧、prompt、action、seed/noise、832×480源分辨率。

必留失败证据：

- [同一checkpoint的20秒完整失败片](long_horizon/original_vs_rgb_visual_W_20s_481f.mp4)：约10秒开始明显雾化，15秒后人物/场景难以辨认；完整后段未裁掉。不能宣称20秒稳定。
- [最新E2：固定D-history，当前A](../reports/stage1_anyflow/real_transition_windows/review_step4/parking_historyD_currentA_comparison.mp4)：左Original权重局部N，中FM-only4，右FM+action4。A方向响应仍在，人物重影没有一致修复。左列**不是**Original整段双向推理。

[最新E2完整报告](../reports/stage1_anyflow/real_transition_windows/FINAL_RESULTS.md)与[主片历史指标](METRICS.md)分别保留，不交叉归因。
