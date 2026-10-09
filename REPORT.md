# 简短实验报告：在 H3-World 中验证 SolarWM 的因果少步生成思路

> 对应面试题的四项交付：方法说明、最小 causal 训练原型与可行性、并排对比 Demo、耗时/显存/质量/连续性指标。更细的协议与诊断见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)，完整问答见 [INTERVIEW_ANSWER.md](INTERVIEW_ANSWER.md)。

## 一句话结论

块因果 + 持久 KV Cache + clean commit 在 33B 的 H3-World 上跑通，加上 RGB 一致的 anchor 后 124 帧画面稳定；但当前展示**没有取得端到端加速**，**动作方向没有恢复**（A−D 0.31～0.45，门槛 1.0）。这是一次可行性验证，不是已经保住动作的少步模型。

新增实现状态：TF-AnyFlow128完整4/8步评测已完成，8步A/D分离度0.760但视觉仍漂移，4步严重雾化；同权重teacher history改善到1.334，但有oracle重置。固定同历史动作干预显示动作仍有独立响应，保真尚未证明。见[128完整结果](reports/stage1_anyflow/05_runtime/parallel_resume68_to128/STEP128_RESULTS.md)、[历史对照](reports/stage1_anyflow/03_anyflow_trials/history128/FINAL_RESULTS.md)及[动作干预](reports/stage1_anyflow/02_causal_diagnostics/counterfactual128/FINAL_RESULTS.md)。同权重有限步/对角条件消融已完成：r=t后段画质明显更完整，但分离度0.662仍未过；有限步映射不足不能全推给Stage2。无新训练、无Stage2。下列表格和会议展示仍是此前的非AnyFlow结果。

## 1. 方法说明

**SolarWM 三个阶段**（仓库命名 / 论文编号）

| 阶段 | 注意力 | 作用 |
|---|---|---|
| Stage0.5 / 论文 Stage 1 | 整段双向 | 双向 flow matching + fused-PRoPE 相机条件；产物是后续初始化，也是 Stage2 里冻结的双向 teacher |
| Stage1 / 论文 Stage 2 | 块因果 | teacher forcing（历史用干净 GT）+ AnyFlow loss（学任意两个噪声水平间的跳转），直接得到少步自回归初始化 |
| Stage2 / 论文 Stage 3 | 块因果 + KV | 学生在自己的 rollout 上训练；冻结 teacher，训练 fake-score critic，以两者 score 差做 DMD 分布匹配；SGF 让梯度流过历史 K/V 的写入 |

**Stage2 为什么能减少采样步数。** 普通 flow 模型学瞬时速度，轨迹是弯的，步长一大就偏离，所以要 30～50 步。Stage1 的 AnyFlow 让模型能直接大步跳转；Stage2 的 DMD 对齐样本**分布**而不是 ODE 轨迹（mode-seeking，少步也锐利），并且学生正是在推理所用的那几步、在自己生成的历史上训练，以减轻误差传播；并不保证误差完全消失。**因果 mask + KV Cache 负责分块执行与历史复用；Stage1 AnyFlow 建立少步能力，Stage2 SGF/DMD 进一步改善自身 rollout 分布上的质量。**

**H3-World 与 SolarWM 因果生成的区别**

| | H3-World 当前流程 | SolarWM 因果生成 |
|---|---|---|
| 视频间注意力 | 全双向 | 块因果（块内双向，块间只看过去） |
| 生成方式 | 一次生成固定 124 帧（37 个 latent） | 逐块流式，长度不限 |
| 步数 | 50 步，每步整段 33B 前向 | 每块约 4 步 |
| KV Cache | 无，每步全部重算 | 有，滑窗限制大小 |
| 控制 | 键盘动作文本，每个 latent 一条 | 相机几何（fused-PRoPE） |

全双向下所有块每一步都在变，没有可复用的 K/V；块因果下已完成块的 K/V 不再变化，才有了 KV Cache。**因果化是因，KV Cache 是果。**

## 2. 最小 causal 训练原型与可行性

- **设计：** 冻结 33B 骨干和已发布的 H3-World LoRA；只训最后 16 个 DiT block 的 QKV rank-8 LoRA（3.44M，约 0.01% 参数）+ 动作 residual（0.77M）。块因果 mask 在 `code/diffsynth_causal.patch`，缓存与调度在 `code/causal/h3_cached.py`。
- **数据与流程：** 单场景 124 帧教师（原始 H3 30 步生成），8 块 × 4 个 sigma。① Stage1-style 干净历史 teacher forcing；② 在线自生成历史，每个 sigma teacher replay + 0.5×端点 MSE，RGB anchor；③ Stage2-lite：student / critic / teacher 共享一个骨干轮换。
- **KV 实现：** 每块 5 个 latent 帧、历史窗口 5 块；去噪期间只读缓存，去噪完成后用 `sigma=0` 的干净 latent 再跑一次，把每层 K/V 写入 CPU 缓存（不能写带噪步）。124 帧 = 8 块 × 8 步 = 64 次带噪前向 + 8 次 commit，replay 误差为 0。
- **可行性证据：** 单张 48 GiB L40 可训，在线更新峰值 32.5 GiB、每次约 9 分钟；训练 loss 0.098→0.062，验证 0.103→0.082；39/124 帧自由 rollout 结构较完整（20 秒失败，见第 3 节）。
- **局限：** 单场景小数据；没做 Stage0.5，Stage1 无 AnyFlow，Stage2 只是 lite（无 SGF）；泛化未验证。


## 3. 并排对比 Demo 与指标

**Demo：** [meeting/annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4](meeting/annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4)（四动作总览：[h3world_rgb_stable_action_grid_124_timed.mp4](meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4)）。左：原始 H3-World 30 步；右：因果原型每块 8 步（RGB anchor）。同首帧、prompt、动作、seed 13、初始噪声、分辨率、124 帧（5.17 s）。

> 早期 fixed-mix 网格（latent anchor）后段会出现重影、人物分解，是我整理会议包时误选的旧协议，已移到 [meeting/diagnostics/legacy_fixed_mix/](meeting/diagnostics/legacy_fixed_mix/) 仅作对照。注意 RGB 版同时更换了 adapter、anchor 和 action routing，改善不能归因于单一组件。

| 指标（W，124 帧，单次记录） | 原始 30 步 | 旧 fixed-mix（崩） | 因果 RGB 版（当前） |
|---|---:|---:|---:|
| 带噪前向次数 | 30 | 64 + 8 commit | 64 + 8 commit |
| 端到端耗时 | 441.5 s | 450.2 s | **767.9 s** |
| 峰值 GPU 显存（allocated） | 39.9 GiB | 39.9 GiB | 30.8 GiB |
| CPU raw KV | 0 | 13.2 GiB | 13.2 GiB |
| 相邻帧 MAD（越低越平滑） | 4.32 | 4.60 | 3.19 |
| 块边界 MAD | 4.97 | 6.29 | 4.12 |
| 末帧人物与场景 | 完整 | A/D 后段重影崩坏 | 结构完整，有模糊/重影 |

- **耗时：** 没有加速。RGB 版 124 帧端到端慢约 1.5–1.7 倍（W/S/A/D：768/673/700/716 s 对 442/445/454/450 s）：前向 64 次对 30 次，且每块要多一次 RGB 解码/重编码。
- **显存：** GPU 峰值各次在 30.8–39.9 GiB 间浮动，另需约 13.2 GiB CPU 缓存。是单次记录、受 offload 状态影响，不据此宣称省显存。
- **质量与连续性：** 用 RGB MAD、块边界 MAD、带符号水平光流和抽帧作代理；MAD 低也可能是模糊或运动变弱。未汇报 FVD/LPIPS/VBench（生成 rollout 无逐帧真值）。
- **动作：** 124 帧 A−D 水平光流分离：原始 2.68，旧 fixed-mix 0.45（A、D 符号正确但幅度弱），当前 RGB 版 0.22（A=−0.78，符号错误）。同一生成状态上因果与教师的 A/D 差异向量范数比 0.878、余弦 −0.015：信号还在，方向被旋转。8 条修复路线均未过门槛。**画面稳定与动作响应是目前未解的取舍**，见 [三列对比](meeting/action_vs_stability/README.md)。

### 更长的视频（同一 RGB checkpoint，未重训）

| 长度 | 方法 | 前向 + commit | 端到端 | 平均 MAD | 块边界 MAD |
|---|---|---:|---:|---:|---:|
| 10.1 s / 243 帧 | 原始 | 30 + 0 | 921 s | 3.80 | 3.88 |
| 10.1 s / 243 帧 | 因果 | 120 + 15 | 1774 s | 2.71 | 3.64 |
| 20.0 s / 481 帧 | 原始 | 30 + 0 | 2406 s | 3.25 | 3.02 |
| 20.0 s / 481 帧 | 因果 | 232 + 29 | 3659 s | 2.76 | 4.67 |

**20 秒因果视频视觉稳定性失败**：约 10 秒开始严重雾化，15 秒后人物和场景难以辨认，末尾退化成模糊色块；10 秒样本后段也有模糊和重影。完整后段均未裁剪，见 [meeting/long_horizon/](meeting/long_horizon/README.md)。CPU KV 在两种长度都稳定在约 13.2 GiB（5 块窗口），说明历史有界；但 RGB anchor 要反复解码已生成前缀，每块耗时不恒定（首块 57–75 s，平均块 114–123 s）。

**修正之前的说法：** 早期曾记录"原始 H3 在 243 帧 OOM，因此因果路径对长视频有显存优势"。后来查明那是原始路径 directed mask 的 eager 构造申请了 25.9 GiB 临时张量，编译同一 mask 构造（语义不变，已做逐元素等价检查）后，481 帧原始 H3 可完整生成。所以**不能再用"原始无法生成长片"作为因果路径的优势**；因果路径证明的是逐块执行和有界历史 KV，不是更快或更省。

## 4. 结论

1. **能因果化：** 工程上可行（块因果 + 持久 KV + clean commit，replay 误差 0）；RGB anchor 让 124 帧结构更完整，但 20 秒严重崩坏，长时稳定性**未通过**。
2. **借鉴是否有效：** 当前展示未证明"长视频效率"收益——展示 checkpoint 没有经过 AnyFlow 少步训练，且 CPU KV 搬运与 RGB anchor 额外开销较大；对"动作控制"不成立——自生成历史下方向被旋转，且稳定版比旧版动作更弱。
3. **下一步：** 先完成 TF-AnyFlow 的真实 33B 训练与 39 帧 A/D、4/8 steps/chunk 对照，验收少步画质和动作控制，再决定多状态动作监督或 Stage2 SGF/DMD。具体 gate 见 [Stage1 计划](STAGE1_ANYFLOW.md)。
