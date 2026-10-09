# Stage1 TF-AnyFlow：实现、验证与验收

[类型III AnyFlow与类型II普通FM的匹配视频](../../meeting/model_types/README.md)：同初始化/容量/两条伪标签/16新增更新，分别评4与8步；不是等计算量，也不是从真实ABot FM48顺序续训。

[数据集视频与 AnyFlow 训练结果导览](../../meeting/DATASET_AND_ANYFLOW.md)：训练脚本名不代表目标；真实 ABot 两轮48更新为 FM，已找到的 AnyFlow 训练为 Original H3 伪标签。

> 本页是AnyFlow实现与历史实验时间线，下面的“正在运行”仅指当时记录。当前研究已冻结，画质/action gate未过；最新结论见[实验报告](../EXPERIMENT_REPORT.md)，所有历史目录的用途见[档案导航](../../reports/stage1_anyflow/README.md)。

2026-10-08 20:15补记：真实数据FM仍在20/48；零更新完整视频已发现第二场景30步generated-history人物分解，GT18同状态动作差分cos=0.017516，无历史首块也低。新图Original纯A/D是弱正控，停车场回归另列。见[完整基线评审](../../reports/stage1_anyflow/01_real_video/real_abot_fm/BASELINE_COMPLETE_REVIEW.md)。以下无新训练/视频等旧措辞仅指当时诊断快照，不能代表当前状态。

2026-10-08最新：受控field对照定位到AnyFlow更新之前的动作几何失配；真实ABot普通causal FM桥接已启动，尚未验收。完整目标A–D的当前状态与前置门槛见[ABCD_STATUS](../../reports/stage1_anyflow/07_protocols/overviews/ABCD_STATUS.md)，不能将数据准备或只读诊断当作Stage1完成。

当前Stage1尚未通过。136训练/4与8步视频及54个双局部指标case已收尾，后段重影未修复。最新同generated-state当前chunk A/D机制诊断12点全部完成：匹配条件下student r=t与Original teacher动作速度差分cosine均值0.0382（范围−0.1023–0.2050），动作幅度为teacher的0.79–3.56倍。时间索引/实际attention路由、KV只读与未来内容负对照检查通过，支持优先调查动作条件函数迁移不足；双向teacher重算历史与student固定KV的结构差异仍需拆分。见[机制结论与归因边界](../../reports/stage1_anyflow/02_causal_diagnostics/generated_action_geometry128/INTERPRETATION.md)、[136完整结果](../../reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/FINAL_RESULTS.md)。该18:09机制诊断当时已结束且没有训练；随后已开展真实ABot的普通FM48桥接和完整零更新视频/36点机制评测，当前结果见页首链接。新的Stage2仍未启动；最多3张项目GPU。

## 参数策略更正（04时段）

官方H3 Stage1的克隆目标时间MLP是**冻结的**，优化器只包含LoRA；本页已完成的16-update pilot额外训练了目标时间MLP。此前把时间MLP更新视为官方必需条件不准确。详见[源码复核与受控修正](STAGE1_PARAMETER_POLICY.md)。冻结分支已通过CPU实际训练检查，并已完成33B总16次更新（GPU2前8次，GPU0后8次；QKV344万参数，time可训练参数0）；6条39帧评测均未通过动作gate，冻结策略没有修复本轮质量。原pilot记录保持不变，公式对照通过不代表优化器策略一致。

## 补齐了什么

此前 `train_pretrained_multichunk.py` 是 causal + clean-history 普通 FM；在线 per-sigma teacher replay 也不是 AnyFlow。新增的是：

1. **目标时间条件**：克隆已加载 H3 时间 MLP，独立 FP32 参数，固定 gate=0.25；DiT 每层调制和输出层同时接收 `(t,r)` 混合条件。
2. **正确的 packed 时间映射**：按 `(当前时间, 目标时间)` 二元组去重。只改变待去噪视频行的目标时间；图片、文本、音频与干净历史保留各自的时间对角线，避免音频/视频同时间时串用目标。
3. **AnyFlow v1.5 loss**：三个 no-grad 前向构造有限差分目标，然后一次带梯度 prediction；对 H3-World 已经是 `noise-clean` 的速度不重复取负号。
4. **batch=1 的类别覆盖**：每个 optimizer step 累积四个样本，2 个 `r=t`、1 个 `r=0`、1 个 `0<r<t`。非 diffusion loss 按本次 logical batch 的原始 diffusion loss 缩放，再乘 Gaussian timestep weight。直接使用官方 batch=1/DP=1 的类别分配会因取整丢掉前两类。
5. **干净历史训练**：用原始 H3 30-step latent 作为 pseudo-GT。每次参数更新后重新建立干净历史 KV；同一 AnyFlow loss 的四次调用共享相同历史、action、anchor、audio、prompt，且不会写 cache。
6. **有限步采样**：benchmark 新增 `--anyflow-adapter`；每步显式传入目标 sigma，执行 `z_r = z_t + (r-t)*u(z_t,t,r)`，轨迹状态使用 FP32。clean commit 使用 `t=r=0`。
7. **独立 checkpoint**：保存 `causal_adapter.pt`、`anyflow_adapter.pt`，以及原有冻结 action adapter；保留 step_00 和中途 checkpoints。FM 老路径仍可独立使用。
8. **协议检查与可复现补丁**：评测拒绝混用不同步数的 time/QKV checkpoint，以及不一致的 anchor、shift、chunk、history、action routing/feedback 配置。导出补丁同时补齐 cached attention 的梯度读取支持；按 action → causal → AnyFlow 顺序可重建，兼容可选 long-video mask patch。

实现位于 [anyflow.py](../../code/causal/anyflow.py)、[训练入口](../../code/causal/train_stage1_anyflow.py)、[benchmark](../../code/causal/benchmark.py)、[DiffSynth patch](../../code/diffsynth_anyflow.patch)。公式来源和与官方配置的差异见 [provenance](../../code/causal/ANYFLOW_PROVENANCE.md)。

## 已完成验证

- 源项目当前完整测试集：`python -m pytest -q tests`，**84 passed / 7.84秒**（独立训练/验证shift合入后，包含full-history与全覆盖LoRA检查）。早期冻结策略、网格与恢复入口合入时为36项；历史收据保留。CPU检查不等同于CUDA画质验证，日志见 `reports/stage1_anyflow/03_anyflow_trials/training_shift12/main_integration_tests.log`。
- 独立复现：从记录的 DiffSynth base 重建临时目录，仅应用提交包的 action、causal、AnyFlow patches，再运行导出的专项测试，**10 passed**；结果保存在 `reports/stage1_anyflow/03_anyflow_trials/early_pilot/export_reproduction.json`。
- AnyFlow loss 和参数梯度逐项对照 SolarWM 官方 H3 实现：普通去噪、终点、一般区间、接近时间边界、非整时间的 diagonal。
- `r=t` 退化到 FM；noise→clean 的有限步更新符号正确；FP32 更新避免 BF16 逐步舍入。
- 真实 H3 小配置验证：目标时间只改变对应行；修改 `r` 确实影响输出；time MLP 和 QKV 都能反传；梯度 checkpoint 可用；读取历史不修改 KV；保存回载输出一致。
- 小型随机 H3 跑完 8 次 AnyFlow optimizer updates（3 chunks，最后一块不足 5 latent frames），独立确认 QKV 和目标时间 MLP 均实际更新。held-out noise 的 weighted objective 从 `2.75052` 到 `2.74759`。这是 **CPU 训练链路 smoke**，不是 33B 画质改进证据。
- 同一小模型与噪声采样流程的 FM control 也跑完 8 updates。两个目标的 total loss 不能直接作画质排名。

原始日志在源项目 `H3-World/outputs/2026-10-07-21/anyflow_stage1/`，最终 AnyFlow CPU 验证子目录为 `cpu_integration_verified/`，本包复制数值日志到 `reports/stage1_anyflow/`。实现过程修复了随机 diagonal 时间在 float32/1000 转换后略越过 `r<=t` 的边界问题，失败 smoke 日志保留，最终验证使用修正后的实现。

## GPU 2 真实单次更新（2026-10-08）

使用 reserve=20 GiB 的 CPU offload 配置，在已有其他进程的 GPU 2 上完成单次真实更新。总可训练参数 19,275,648，QKV 和 target-time MLP 均确认实际变化；单更新 161.8 秒，含准备和前后验证共 499.7 秒，allocated peak=17,221.8 MiB。日志见 `reports/stage1_anyflow/03_anyflow_trials/early_pilot/pretrained_gpu2_smoke_training.json`，运行环境见同目录 `pretrained_gpu2_runtime.json`。

A/chunk2 held-out 加权验证 loss：0.117473 → 0.118669；高噪声 endpoint raw loss：133.788 → 42.081。整体没有单调改善，不能用其中一项下降宣称视频质量提高。

回载推理发现并修复 FP32 轨迹与 BF16 条件的类型冲突；只在模型调用处转换计算输入，轨迹累积继续使用 FP32。新增混合精度多步/clean commit 与 RGB anchor dtype 测试，AnyFlow 专项共 12 passed。修复后的回载已完成：39 帧、12 noisy forwards + 3 commits、E2E 194.1 秒、GPU allocated peak=16,247.2 MiB、CPU KV=6,484.1 MiB。A 水平光流为 -1.296，contact sheet 后半段明显重影与雾化；这是 1-update smoke，未通过质量 gate。

正式队列已在 GPU 2 启动，计划见 `reports/stage1_anyflow/03_anyflow_trials/early_pilot/gpu2_pilot_plan.json`。源运行目录为 `H3-World/outputs/2026-10-08-02/stage1_anyflow39_pilot/`，`run.json` 记录当前阶段。依次完成 16-update AnyFlow、4/8-step A/D、FM 对照、中途 checkpoint 学习曲线、clean-history 诊断；任何失败自动停止。

## 16-update 实测（4/8-step A/D已完成）

训练完成不等于Stage1验收：训练总计3043.0秒，allocated peak 17,439.4 MiB，time MLP和QKV均实际更新。固定验证noise的A/D total loss略降，但高噪声endpoint raw loss均上升（A 133.788→144.185；D 18.083→19.440）。完整数据在[训练日志](../../reports/stage1_anyflow/03_anyflow_trials/early_pilot/pretrained_gpu2_anyflow16_training.json)。

4-step/chunk generated-history：A=-1.3153、D=-1.2244、A-D=-0.0909；两条视频后段重影/雾化，动作与视觉gate失败。每条均完整39帧，12 noisy forwards+3 clean commits。输入逐张量与对应原始H3一致。8-step A=-1.1709、D=-1.4752、A-D=0.3043，画面比4-step完整但动作gate仍失败。FM16的4/8-step分离度为0.2338/0.3149；AnyFlow4 step00/04/08/12/16没有持续改善。clean-history A-D=0.5135仍失败；[动态报告快照](../../reports/stage1_anyflow/03_anyflow_trials/pilot_snapshot/REPORT.md)只列已生成结果。旧原始H3的offload reserve未知，时间和GPU peak仅作本机运行记录，不能据此计算公平加速比。

可选[位置契约helper](../../code/causal/position_contract.py)解决未来action句长通过RoPE原点影响过去的结构性问题，小型H3的FM/AnyFlow测试2 passed；固定A/D输入是数值no-op。**尚未接入当前benchmark/trainer，也不是已证实的视觉失败原因。**

[三列诊断视频](../../reports/stage1_anyflow/03_anyflow_trials/pilot_snapshot/anyflow16_AD_original_4step_8step_diagnostic.mp4)与[人工观察记录](../../reports/stage1_anyflow/03_anyflow_trials/pilot_snapshot/VISUAL_REVIEW.md)保留全部39帧，明确标记未通过，不替换旧会议demo。

## 已完成pilot的33B实验协议（trainable-time变体）

先用 39 RGB frames / 12 latent frames / 5+5+2 chunks；chunk size=5，history=5，seed=13，shift=2.22，causal action prefix，feedback ON，CPU raw KV，RGB dual anchor。训练使用 **clean teacher history**，主评测使用 **generated history**，两者分别报告。

初始化沿用当前 RGB visual QKV 和 fixed-mix action residual，只训练 QKV 与新增 target-time MLP，action residual 冻结。先验证一个真实 update 的显存、有限梯度、checkpoint 回载和推理；通过后再跑 16 updates、每 4 步保存。另跑相同 QKV 初始化和数据/噪声流程的普通 FM control。两者训练算力和参数量并不相同：AnyFlow 额外有 target-time MLP，每个样本 4 次前向；必须明确报告，不能声称是等算力对照。

从源项目 `H3-World/` 运行（先选择空闲 GPU；`CUDA_VISIBLE_DEVICES` 映射后 `cuda:0` 指向所选卡）：

```bash
python code/causal/train_stage1_anyflow.py \
  --teacher-dir outputs/2026-10-02-03/action_A_teacher_39 outputs/2026-10-02-03/action_D_teacher_39 \
  --actions A D --objective anyflow --train-target-time --steps 16 --checkpoint-every 4 \
  --causal-adapter outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/causal_adapter.pt \
  --action-adapter outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/action_adapter.pt \
  --anchor-mode rgb --flow-shift 2.22 --action-prefix-mode causal --action-feedback \
  --out-dir outputs/YYYY-MM-DD-HH/stage1_anyflow39
```

上面的 `--train-target-time` 用于复现旧pilot变体。当前默认及新的官方参数策略对照应改为 `--no-train-target-time`，只优化QKV；初始时间MLP仍从同一base克隆。二者必须用不同输出目录。

第一次真实 smoke 把 `--steps` 改为 `1`、`--checkpoint-every` 改为 `1`，使用单独输出目录。普通 FM control 使用 `--objective fm` 和另一个新目录，其他参数保持一致。

AnyFlow checkpoint 评测（A、D 分别运行；分别测 4 和 8 steps/chunk）：

```bash
python code/causal/benchmark.py --modes cached --num-frames 39 --steps 4 --seed 13 \
  --flow-shift 2.22 --chunk-frames 5 --history-chunks 5 --cache-device cpu \
  --action-preset A --action-prefix-mode causal --action-feedback \
  --anchor-mode dynamic_last_frame_rgb_dual \
  --causal-adapter outputs/YYYY-MM-DD-HH/stage1_anyflow39/causal_adapter.pt \
  --causal-action-adapter outputs/YYYY-MM-DD-HH/stage1_anyflow39/action_adapter.pt \
  --anyflow-adapter outputs/YYYY-MM-DD-HH/stage1_anyflow39/anyflow_adapter.pt \
  --out-dir outputs/YYYY-MM-DD-HH/stage1_anyflow39/eval4/A --save-latents
```

官方均匀网格另加 `--anyflow-sigma-grid uniform`（4-step为1→.75→.5→.25→0）；训练的 `--flow-shift 2.22` 仍保留并用于协议检查。默认 `native` 精确保留旧采样网格。

FM control 不传 `--anyflow-adapter`；其余评测一致。另测 step_00 的 AnyFlow 模块，区分目标时间条件初始化造成的变化与训练带来的变化。clean-history 诊断另加 `--history-source clean --teacher-latents <对应动作的 baseline_latents.pt>`，不要混作主 rollout。

新训练保存 `trainer_state.pt` 后，可保留原训练参数，在新输出目录使用 `--resume-from <旧run/step_16> --steps 32` 续训到总共32次更新；不能只加载adapter并声称恢复了Adam。需要同时保留原teacher目录和初始化参数，配置变化会被拒绝。旧pilot不含optimizer/RNG状态，只能作为权重热启动；CPU精确续训验证见[参数策略与恢复证据](STAGE1_PARAMETER_POLICY.md)。

完整要求和逐项状态见 [Stage1 验收清单](STAGE1_ACCEPTANCE.md)。

## Stage1 何时可以进入下一阶段

1. 实现 gate：公式/梯度对照、时间条件、cache、保存回载通过；33B真实训练无NaN/OOM，QKV实际更新。与官方对齐的分支必须验证时间MLP严格冻结且修改目标时间仍改变输出；旧trainable-time变体单独记录。
2. 少步 gate：在完全一致的首帧、prompt、动作、noise 下，比较 FM 8-step、AnyFlow step_00、AnyFlow 4/8-step 和 original H3 30-step；单独记录每类 loss、方向/端点误差、耗时/显存和完整视频。训练 total loss 下降不构成通过。
3. 动作与视觉 gate：39 帧 A>0、D<0、A-D>1.0，同时人物和场景不分解；光流必须与完整视频一起解释。至少再验证一个未用于选择 checkpoint 的 noise seed。未通过则继续定位 Stage1，不能用 DMD 掩盖。
4. 短片通过后再测 action switching 和 124 帧 W/S/A/D。只有同时报告少步质量、动作响应和 generated-history 行为，才决定 Stage2 实验。

AnyFlow 补齐的是有限步映射训练；它不保证单场景/低容量适配会恢复 H3 原始 action geometry，也不能消除所有历史误差。本实现保留 detached clean-history cache，训练数据是有限 teacher 轨迹，均不同于官方大规模两流训练。上述 gate 尚未完成时，状态必须保持“Stage1 待验收”。

## 安装和 CPU 检查

先应用既有 H3 action/causal patch，再从 submission 根目录执行：

```bash
git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_anyflow.patch
python -m pip install pytest
python -m pytest -q tests/test_anyflow.py
python code/causal/train_stage1_anyflow.py --smoke --steps 8 --lr 0.001 \
  --out-dir outputs/anyflow_cpu_smoke
```

与官方公式的 oracle 测试需要本项目根目录旁的 `SolarWM/src` checkout，参考版本为 `ce1da4e7705391eda8eeda6016c0fd3f614b975e`；运行训练与采样无需导入 SolarWM，所需数学 primitives 已附并保留 Apache-2.0 归属。

全覆盖rank8候选已完成CPU预检查并条件等待：保留相同初始函数和数值/数据协议，按官方312个投影范围训练LoRA；详见[候选范围、参数化差异与限制](../../reports/stage1_anyflow/03_anyflow_trials/fullscope_candidate/README.md)。真实GPU fit与画质尚待验证，不能把这个候选称为已完成的完整官方Stage1。
