# Code map

- causal/h3_cached.py：chunk-wise attention、persistent raw KV、clean commit、RGB/image anchor 和 action feedback。
- causal/benchmark.py：original/recompute/cached causal inference 与性能/视频指标。
- causal/pretrained_lora.py：实验 adapter 的安装、保存和加载。
- causal/train_online_selfrollout.py：generated-history per-sigma teacher replay、paired action diagnostic。
- causal/stage2_lite_dmd.py：shared-backbone student/critic/teacher Stage2-lite chain。
- causal/evaluate_action_control.py：signed horizontal optical-flow proxy。
- causal/probe_action_geometry.py、probe_action_routing.py：teacher/causal score geometry 和 routing audit。
- abot/：最小 H3 action preset 和 inference loader。
- diffsynth_h3_action.patch：H3-World action-directed attention patch。
- diffsynth_causal.patch：causal/cache 相关 DiffSynth 变更。

所有 .py 文件均为源代码；提交时已排除 __pycache__、latent、conditioning 和运行日志。
- diffsynth_long_video_mask.patch：可选编译原始 H3 directed action mask 构造，避免长视频的 eager 二次方临时张量；不改变 mask 语义。
- causal/check_long_mask.py：eager/compiled mask 及 action assignment 变更的逐元素等价检查。
- causal/anyflow.py、anyflow_reference.py：目标时间条件与 SolarWM v1.5 有限差分目标；来源及简化边界见 causal/ANYFLOW_PROVENANCE.md。
- causal/train_stage1_anyflow.py：clean-history TF-AnyFlow 和普通 FM control；physical batch 1 / logical batch 4；独立 target-time checkpoint。
- diffsynth_anyflow.patch：按 (t,r) 构造 packed 时间索引，接入 DiT 时间调制，并补齐 detached-cache 梯度读取与已有 hidden action residual 的调用；benchmark 的 --anyflow-adapter 使用目标时间有限步采样。
- diffsynth_native_fp32.patch：在AnyFlow补丁后应用，支持显式`--precision-profile h3_fp32`，保留FP32输入扰动和时间条件；`causal/h3_precision.py`恢复原生FP32边界权重并校验checkpoint精度来源。默认legacy，数值策略改变不代表效果已通过。

- causal/anyflow_sampling.py：明确区分native与uniform推理sigma网格，不混淆训练shift。
- causal/training_state.py：Adam、RNG与adapter哈希绑定的续训校验；`--resume-from`和目标总更新数。
- causal/stage1_lora.py：可选的全部主块和refiner Q/K/V/out/FFN LoRA。保留旧visual/action初始化，新增B零初始化bank；Q/K/V使用独立低秩因子。默认仍为tail QKV，显式`--adapter-scope all_qkvo_ffn`才启用。
- causal/stage1_protocol.py：全覆盖FM/AnyFlow评测的checkpoint与训练协议检查，拒绝遗漏bank或混用不同步数、精度、action、anchor配置。
- causal/shared_h3_roles.py：冻结H3底座上的student / Original teacher / independent critic角色；使用独立LoRA参数与enabled flags，保护保留的FMBS计算图。包含关闭并恢复AnyFlow/旧visual residual及错误角色backward检查；不替caller选择attention、conditions或KV。见[6项tiny-H3图安全验证](../reports/stage1_anyflow/06_stage2_preparation/shared_h3_roles/README.md)，尚未用于新的33B DMD效果实验。
- causal/dmd.py：H3 noise-minus-clean下的DMD方向、detached重加噪及fake-score FM目标；接收保留生成图的endpoint，支持完整FMBS的J^T g。见[7项符号/官方对照/H3梯度检查](../reports/stage1_anyflow/06_stage2_preparation/dmd_gradient/README.md)，没有新的33B Stage2画质结论。
- train_stage1_anyflow.py新增`--training-weight-shift`、`--validation-weight-shift`，与sigma采样density解耦；不指定时精确保留旧耦合行为。新[真实FM密度对照](../reports/stage1_anyflow/01_real_video/fm_density_control/README.md)固定权重shift12，只把采样shift12改成2.22；48更新预算独立封顶，训练与评测已收尾，质量gate未过；以最终报告为准。

全覆盖checkpoint额外包含`stage1_lora.pt`，推理必须传`--stage1-lora`。AnyFlow仍需要其matching `anyflow_adapter.pt`；FM不传AnyFlow模块。rank=alpha=8共有43,237,376训练参数，实际效果仍待验收，不能视为官方rank384 Stage1复现。

- causal/clean_history_graph.py：可选带梯度的clean-history KV预填充；每块读取不可变快照，checkpoint重算时不重复捕获KV。训练用`--history-gradient-mode full`，默认detached，推理协议不变。真实33B探针通过，画质待验证。

`causal/fmbs.py`提供AnyFlow三段Flow Map Backward Simulation与H3 callback接入。历史KV列表快照固定、当前三段状态保留梯度；这是生成端原语，尚不是teacher/critic/DMD trainer。实际H3随机小模型的CPU检查见[集成报告](../reports/stage1_anyflow/06_stage2_preparation/h3_fmbs_integration/README.md)。

最终提交运行入口：`scripts/prepare_runtime.py`重建源码并验证显式外部依赖；`scripts/verify_inference.py`核对主片adapter哈希与39f推理输出。`tests/test_h3_cached.py`已随包提供。见[REPRODUCE](../REPRODUCE.md)。
