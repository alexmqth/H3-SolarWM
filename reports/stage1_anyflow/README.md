# 实验档案导航：为什么这里有这么多目录？

本目录有 **51个一级子目录**。它最初用于Stage1/AnyFlow，后来沿用了目录名，把普通causal FM、真实ABot训练、动作机制诊断、数值检查和DMD工程准备也归档在这里。**目录名不等于实验都使用AnyFlow；51个目录也不等于51个独立模型、51轮正式训练或51个待执行任务。**

目录多的原因是保存了不同协议的原始证据，包括中间checkpoint的评测、失败尝试、只读探针、控制器收据与冻结源码。它有助于追溯，但原先缺少导航，不适合作为面试时逐一浏览的入口。本页只整理分类，不搬动或改写冻结结果。

当前终态：研究实验已冻结，E2两臂各4更新无一致收益；局部动作＋画质联合验收仍未过。不扩训、不自动重启AnyFlow或Stage2。旧报告里“正在运行”只代表当时快照；当前结论以[最终实验报告](../../EXPERIMENT_REPORT.md)、[计划](../../NEXT_PLAN.md)和[E2冻结收据](real_transition_windows/FREEZE.json)为准。

## 只想看成果，从这里进入

- **会议主Demo与5分钟讲稿：** [meeting/README](../../meeting/README.md)。主片来自旧RGB checkpoint，不是最新E2/AnyFlow。
- **昨晚真实视频训练：** [集中展示页](../../meeting/OVERNIGHT_PROGRESS.md)。包含GT/Original/FM0/FM48四列视频和generated-history失败。
- **最新研究结论：** [EXPERIMENT_REPORT](../../EXPERIMENT_REPORT.md)。
- **实际复现是否通过：** [独立环境验收](../final_acceptance/README.md)。

## 当前最值得看的五个目录

| 目录 | 它在回答什么问题 | 直接读哪份结果 |
|---|---|---|
| `real_abot_fm/` | 换成真实ABot录制数据，做48次普通causal FM更新，能否恢复局部生成与动作？ | [完整验收](real_abot_fm/FM48_COMPLETE_REVIEW.md)：没有恢复 |
| `fm_density_control/` | 另做48次更新，只调整训练噪声采样分布，低噪声覆盖增加是否有效？ | [最终结果](fm_density_control/FINAL_RESULTS.md)：未通过 |
| `history_conditioning/` | 同一历史/当前动作下，clean历史与同sigma加噪历史有什么区别？ | [视频结果](history_conditioning/VIDEO_RESULTS.md)：后续窗A/D方向改善，人物仍重影；零训练 |
| `real_transition_windows/` | 同初始化、各4更新，FM＋动作后果监督是否优于FM-only？ | [最终结果](real_transition_windows/FINAL_RESULTS.md)：没有一致额外收益，冻结 |
| `local_topology/` | 如何保留Original H3的局部动作信息流，同时禁止未来信息？ | [拓扑实验](local_topology/README.md)：因果/执行检查与画质验收分别看 |

## 其余目录是什么

- **训练/采样变体：** 如`pilot_snapshot`、`full_history_duration64`、`interval_consistency_candidate`。记录当时的AnyFlow/FM对照、历史梯度、训练预算或采样方式。有的是同一条训练链的不同checkpoint评测，并非新模型。
- **数值和工程检查：** 如`precision_probe`、`time_contract`、`gradient_repeatability`。检查dtype、时间条件、梯度或输出是否正确，不是画质结果。
- **动作机制诊断：** 如`first_chunk_routes`、`generated_action_geometry128`。固定状态只换动作或路由，定位动作差分为何失配；不等于训练。
- **DMD准备：** `shared_h3_roles`、`h3_fmbs_integration`、`dmd_gradient`等验证共享底座角色、生成Jacobian及DMD方向；主要是小H3/CPU工程证据，不能算完整33B Stage2结果。
- **运行组织：** `gpu0_migration`、`parallel_resume68_to128`等保存历史设备迁移/并行/续训收据，防止重复启动或错误resume；不是当前待办。

`128/136`等通常标记当时被检查的checkpoint或训练更新预算，`shift12`涉及时间/噪声分布，`window12`涉及latent窗口。不要把这些名字理解成独立模型版本或生成视频秒数；精确定义仍以各目录协议为准。

## 一个目录里面为什么又有很多文件？

常见文件的职责分别是：

| 文件或子目录 | 用途 |
|---|---|
| `README.md`、`FINAL_RESULTS.md`、`*REVIEW.md` | 协议、结果与人工观察；优先看最终结果 |
| `*.json`、`*.csv`、`*.log` | loss、时间/显存、动作指标、配置/哈希、运行收据 |
| `report/`、`review*/`、`*.mp4`、`*.jpg` | 对比视频、全帧检查图、原尺寸人物细节；有源视频和拼接展示两个层次 |
| `geometry/`、`probe*/` | 固定状态动作差分、路由、噪声等诊断 |
| `runtime/`、`runtime_manifest.json` | 实验当时的源码副本或源码哈希，用来证明运行期间代码未被修改 |

其中`local_topology/runtime/`保存了源码快照，文件数量较多；`real_transition_windows/`还保留各单条生成、两列/三列对照和帧图，所以最占空间。这些都不是33B模型权重。

原始视频数据、编码latent、大模型及这几轮新训练的adapter/optimizer仍在本机实验目录，没有随这些报告全量上传。正式提交/会议应从摘要和精选视频进入，只有追问某个结论时再下钻原始证据。

## 全部51个目录的分类索引

以下仅为阅读分组，不改变原有路径；各组不表示相互独立的实验或已通过验收。

### 主要真实数据训练结果（3）

- [fm_density_control](fm_density_control/FINAL_RESULTS.md)
- [real_abot_fm](real_abot_fm/README.md)
- [real_transition_windows](real_transition_windows/FINAL_RESULTS.md)

### 动作信息流、局部拓扑与历史条件诊断（9）

- [coarse_window12](coarse_window12/FINAL_RESULTS.md)
- [counterfactual128](counterfactual128/FINAL_RESULTS.md)
- [current_prefix_candidate](current_prefix_candidate/README.md)
- [field_factorization](field_factorization/README.md)
- [first_chunk_routes](first_chunk_routes/README.md)
- [fm_noise_audit](fm_noise_audit/README.md)
- [generated_action_geometry128](generated_action_geometry128/README.md)
- [history_conditioning](history_conditioning/README.md)
- [local_topology](local_topology/README.md)

### AnyFlow及普通FM对照、训练预算和采样诊断（历史）（21）

- [diagonal_inference128](diagonal_inference128/FINAL_RESULTS.md)
- [dual_metric136](dual_metric136/README.md)
- [duration32_snapshot](duration32_snapshot/README.md)
- [duration_response](duration_response/RESULTS.md)
- [finite_interval_probe128](finite_interval_probe128/README.md)
- [finite_interval_refinement128](finite_interval_refinement128/README.md)
- [fm_full_history_shift12_32](fm_full_history_shift12_32/FINAL_RESULTS.md)
- [frozen_time_snapshot](frozen_time_snapshot/README.md)
- [full_history_candidate](full_history_candidate/FINAL_RESULTS.md)
- [full_history_duration64](full_history_duration64/FINAL_RESULTS.md)
- [full_history_step04](full_history_step04/FINAL_RESULTS.md)
- [fullscope_candidate](fullscope_candidate/FINAL_RESULTS.md)
- [fullscope_fm_control](fullscope_fm_control/FINAL_RESULTS.md)
- [history128](history128/FINAL_RESULTS.md)
- [interval_consistency_candidate](interval_consistency_candidate/FINAL_RESULTS.md)
- [pilot_snapshot](pilot_snapshot/REPORT.md)
- [shift12_duration64](shift12_duration64/FINAL_RESULTS.md)
- [shift12_history_diagnostic](shift12_history_diagnostic/FINAL_RESULTS.md)
- [shift12_step32_4step](shift12_step32_4step/README.md)
- [training_shift12](training_shift12/FINAL_RESULTS.md)
- [uniform_snapshot](uniform_snapshot/README.md)

### 精度、梯度与数值策略（8）

- [adaptive_weight_audit](adaptive_weight_audit/README.md)
- [clean_history_gradient](clean_history_gradient/README.md)
- [diagonal_compute_probe](diagonal_compute_probe/README.md)
- [fp32_inputs](fp32_inputs/README.md)
- [gradient_repeatability](gradient_repeatability/README.md)
- [native_fp32_candidate](native_fp32_candidate/FINAL_RESULTS.md)
- [precision_probe](precision_probe/README.md)
- [time_contract](time_contract/README.md)

### 历史GPU迁移、并行检查与续训记录（4）

- [gpu0_migration](gpu0_migration/README.md)
- [parallel_duration128](parallel_duration128/README.md)
- [parallel_resume68_to128](parallel_resume68_to128/README.md)
- [sample_parallel_probe](sample_parallel_probe/RESULTS.md)

### DMD与生成梯度的工程准备（4）

- [cd_method_audit](cd_method_audit/README.md)
- [dmd_gradient](dmd_gradient/README.md)
- [h3_fmbs_integration](h3_fmbs_integration/README.md)
- [shared_h3_roles](shared_h3_roles/README.md)

### 历史计划与协议（2）

- [continuation_plan](continuation_plan/README.md)
- [three_experiments](three_experiments)

