# 实验日志（精简版）

按时间顺序只保留"做了什么、得到什么、为什么走向下一步"。逐轮的原始流水账（含大量运行中状态和过期计划）已经精简掉，完整版可在 git 历史中找到。所有 flow 数值是水平 Farneback 光流代理；"A−D" 是 A 与 D 的水平光流之差，原始 H3 教师在 39 帧下为 2.023。

## 阶段 0：环境与基线（09-29 ~ 09-30）

- 搭好 H3-World（固定 DiffSynth 版本 + 动作 patch）与 SolarWM 两个独立环境；单卡 48 GiB L40 需要 `vram_limit` 才能跑 33B。
- 原始基线（124 帧，832×480）：50 步约 12 分 12 秒，峰值 42,179 MiB；30 步采样约 375 s。
- **仅改 mask 的因果推理**（无训练）：采样约快 11%，显存无变化（43,607 vs 43,583 MiB），帧间 MAD 从 3.40 恶化到 5.36 → 直接屏蔽未来、不做因果训练有质量代价。
- **真正的分块 + raw-KV rollout**（124 帧，4 步/块）：32 次带噪前向 + 8 次 clean commit，采样 231.7 s（原始 30 步 374.6 s，4 步 55.1 s），CPU KV 13.2 GiB。能跑通，但视频明显退化（未训练）。
- 22 帧 cached vs 全历史重算：cached 反而慢约 19%（CPU 缓存搬运），最终 latent 相对 L2 差异 1.533%（混合后端）/ 0.502%（统一 SDPA）。GPU 上放 KV 会在逐层加载权重时 OOM。

## 阶段 1：Stage1-style 因果适配（09-30 ~ 10-01）

- 只训最后若干 block 的 rank-8 QKV LoRA，clean-history teacher forcing。训练 loss 稳定下降约 20%、replay 误差 0，但自由 rollout 近乎静止或后段漂移 → loss 下降不等于 rollout 好。
- **抓到 benchmark 的 bug**：`scheduler.step` 在 timestep 循环外，"8 步"实际只前进一步。修正后之前结果作废。
- 尾帧作为第二 anchor（双 anchor）、scheduler `shift=2.22` 对齐、RGB-prefix anchor：有改善但约第 30 帧后仍有重影；latent anchor 与 H3 原生图像编码语义不同。
- **抓到训练端 bug**：尾部 4 个 block 复用了同一 `layer_index`，读了错误层的历史 K/V（`replay_tail` 修复）。
- 负结果：给 clean history 加高斯噪声、静态混入已有 generated rollout，均加剧漂移。
- 扩大适配容量 tail4 → tail8 → tail16，tail16 消除了 chunk 边界的硬切换，成为当时的主 demo；243 帧因果可生成。（当时记录的“原始 30 步在 243 帧 OOM”后来查明是 eager mask 的实现问题，已撤回，见 REPORT.md 第 3 节。）

## 阶段 2：动作保真度的拆解（10-02 ~ 10-03）

39 帧、30 步/块的 A−D：

| 条件 | A−D |
|---|---:|
| 原始 H3 教师 | 2.023 |
| 因果，无适配，clean history | 0.956 |
| 因果，无适配，generated history | 0.015 |

→ 因果 mask 损失一半，自生成历史几乎清零。`action_prefix_mode=all` 上界也不能恢复。用 generated-history 做 scheduled sampling：fixed-mix 0.5 在 39 帧最好（A−D=1.003，30 步/块）；换到 124 帧、8 步/块只有 **0.453**，成为正式 124 帧 W/S/A/D grid（`meeting/diagnostics/legacy_fixed_mix/`）。

## 阶段 3：在线 self-rollout 与动作路径适配（10-03 ~ 10-04）

固定协议（39 帧、3 块、8 步/块、seed 13），A−D：

| 实验 | A−D |
|---|---:|
| fixed-mix 基线（feedback on） | 0.756 |
| online per-sigma teacher replay（4 步） | 0.798 |
| + 配对 A/D 差异损失 N1 / N2（4/16 步） | 0.769 / 0.74~0.80 |
| action-prefix residual | 0.772 |
| 改原始 H3 action LoRA tail8 | 0.782 |
| 端点 latent 监督 | 0.746 |
| 最小 two-pass replay | 0.754 |

内部 `L_dir`/`L_mag` 持续下降，但 rollout 方向不恢复。

## 阶段 4：Stage2-lite（10-05）

共享 33B 骨干，轮换 student / critic / teacher 三个小 adapter：student self-rollout → critic 拟合 → frozen teacher → DMD surrogate。4 轮更新 A−D=0.773（A=+0.005），人物 15~30 帧后雾化。结论：链路可运行（峰值 39.4 GiB，无 NaN/OOM），但单 chunk、单 sigma 覆盖太窄。

## 阶段 5：视觉修复与动作几何诊断（10-06）

- **视觉**：人物分解的根因是 anchor 协议不一致（训练用 RGB 解码重编码，推理用 latent patchify）。改为 RGB 双 anchor + tail16 visual QKV + 原始 H3 块端点监督：39 帧 A/D、124 帧 W/A/D 均稳定。W 长片采样 709 s，GPU 峰值 31,570 MiB，CPU KV 13.19 GiB。
- 把修复后的 visual adapter 接回真正的 Stage2-lite（多 chunk、四个 sigma）：A−D=0.313，视觉稳定但动作未恢复。
- **几何诊断**：同一 generated state 上，教师 A/D 差异仅占速度范数约 2%；因果学生的差异范数是教师的约 6.9 倍且余弦 ≈ −0.009；冻结因果与教师幅度比 0.878、余弦 −0.015。结论：信号存在，方向被旋转。
- 动作路径适配（tail4 action-QKV 4 轮、action-prefix、released H3 LoRA、per-action gain 训练、端点监督含 own-history 修正）全部未过 gate，A−D 在 −0.12 ~ 0.83。
- 决定：不再做单场景单状态的 gain / anchor / solver / 端点权重扫描；不再生成新的 124 帧动作 grid。正式 124 帧交付后来改为 RGB 版（见阶段 6）；旧 fixed-mix grid 因后段崩坏已归档为对照。

## 一句话总结

分块因果 + 持久 KV + clean commit + RGB anchor 在工程和视觉上跑通；动作几何在自生成历史下被旋转，需要多状态/多 seed 的动作监督或完整 Stage2 级的 rollout 分布匹配。

## 阶段 6：纠错、动作/视觉取舍与 10/20 秒长视频（10-07）

- **纠错：** 第一版会议包误把旧 fixed-mix 网格（latent anchor、无 tail16 visual adapter、own prefix、无 feedback）当主视频，后段重影崩坏。主视频改为 RGB 版（`meeting/annotated/h3world_rgb_stable_*`），旧版归档到 `meeting/diagnostics/legacy_fixed_mix/`。切换同时改了 adapter、anchor、routing，改善不能归因于单一组件。
- **取舍：** 旧版动作较强（A−D 0.453，A 正 D 负）但画面崩；RGB 版画面稳但 A−D 仅 0.223、A 符号为负。三列对照见 `meeting/action_vs_stability/`。
- **长视频（同一 RGB checkpoint，未重训）：** 243 帧（120 + 15 次前向）与 481 帧（232 + 29）均完整生成并与匹配输入的原始 30 步对照。**20 秒严重失败**：约 10 秒起雾化重影，15 秒后难以辨认；10 秒后段也有模糊。CPU KV 稳定在 13.2 GiB，说明历史有界。端到端约为原始的 1.5–1.9 倍。
- **撤回早期结论：** 原始 H3 在长视频 OOM 是 directed mask eager 构造申请 25.9 GiB 临时张量所致；编译同一 mask 构造（语义不变，逐元素等价）后 481 帧原始可完整生成。因果路径不再有"原始无法生成长片"这一优势。
