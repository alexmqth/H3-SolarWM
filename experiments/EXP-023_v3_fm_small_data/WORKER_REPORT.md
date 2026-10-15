# EXP-023/v1 Worker 报告：真实视频普通 FM 小数据拟合

2026-10-15 HKT。执行状态：**已完成，待 Judge 验收**。能力结论：固定状态 FM 拟合略有改善；匹配视频没有提供清楚的动作或视觉能力增益证据。训练成功不等于 V3 质量验收通过。

## 研究问题与冻结协议

本轮检验 32 条真实视频、128 次普通 Flow Matching 更新，能否在不改变 V3 严格 chunk-causal / persistent raw KV 推理协议的前提下，改善 C2 局部续写，并把收益带到自生成 C1 历史。依据 [任务书](taskbook_v1.md) 和 [补充01–05](ADDENDUM_05_TRAIN_HOOK_RECOVERY.md)。实际输入为 32 train + 2 dev，共 34 条各 56 RGB 帧、832×480、24 FPS；全部从冻结源视频索引直接提取 RGB，分别编码 I0 与 17 latent，未复用旧 H.264 latent。两条 pilot 的独立 prefix12 与 full17 前缀逐值相同。[选样](SELECTION.json) · [源与编码审计](ENCODED_DATASET_AUDIT.json) · [输入 SHA](TRAIN_SOURCE_MANIFEST.json)。dev 来自已观察过的 episode，不是盲测集。

原始 H3 与 released action LoRA 冻结，新增零增量初始化 tail8 QKV rank8 adapter，1,720,320 参数。仅训练 C2 的 5 latent；C1 使用 GT clean commit，每次更新重建 raw KV。Single I0、native time、Global RoPE、own-action/current-prefix/action feedback、h3_fp32/SDPA 均保持冻结。AdamW、lr 1e-4、batch 1，32 样本×4 epoch、seed 13；目标是普通 epsilon-clean FM，没有 AnyFlow target-time、DMD 或额外动作 loss。前后评测均为 FM-8NFE/chunk，12+5 latent、56 RGB 帧。

## 执行与协议验证

- checkpoint0 与冻结入口：velocity 最大绝对误差为 0，50 层的 150 项 K/V/RoPE 张量逐值一致。[入口回归](FROZEN_ENTRY_REGRESSION.json)。
- checkpoint backward 的路由作用域经过 [CPU 回归](HOOK_LIFECYCLE_CPU_V2.json) 与首次真实 33B backward 门控。实际重算层 49→42 均保持 `current_prefix`；历史 CPU KV 不变，冻结 base/action LoRA 无梯度，adapter 梯度有限非零。[真实门控](GRADIENT_PROTOCOL_CHECK.json)。这修正了 EXP-022 旧 forward-only context 的梯度协议缺陷，但不追溯改写旧实验结论。
- 单 GPU0 完成 128 backward / 128 optimizer updates / 256 训练顶层 DiT。保存 [checkpoint_008](checkpoint_008.json)、[032](checkpoint_032.json)、[064](checkpoint_064.json) 与 [128](checkpoint_128.json)；128 文件 SHA256 `735b2dc04f5c8ca6f581d142220db7f74eeed74443503c94880a325f3128bcde`，本地重算匹配，包含 adapter、optimizer、CPU/CUDA/数据顺序/噪声 RNG，数据顺序位置 128。大 checkpoint 与编码 tensor 留在本地，不建议纳入 Git。
- 训练阶段 2650.296 秒，峰值 GPU allocated 24.540 GiB、reserved 28.252 GiB，每步 CPU raw KV 6,799,104,000 bytes。训练逐更新原始 loss/σ/clip/梯度在 [TRAIN_UPDATES.json](TRAIN_UPDATES.json)；逐步 σ 不同，不能把 loss 起伏解释为学习曲线。按四个 32-update 段的 loss 均值分别为 0.1774、0.2357、0.1928、0.2334；σ 采样差异很大。训练阶段账本为 [TRAIN_RETRY_BUDGET.json](TRAIN_RETRY_BUDGET.json)。

## 固定前后评测

四条 GT-history 视频：展示的前 39 帧为 GT，只生成后 17 帧。两条 dev self-history 视频：C1 与 C2 都由各 checkpoint 从同 I0、动作、seed/噪声自行生成，前后 checkpoint 的 C1 状态因此不相同。两个 checkpoint 各 6 条视频，共 12 条 56 帧/24 FPS；另做四 clip × σ=0.25/0.75 的同噪声固定状态 FM probe。[原始前视频](eval_before/) · [原始后视频](eval_after/) · [6组三列可播放对照与关键帧](paired_gallery/) · [媒体完整性](FINAL_VIDEO_AUDIT.json) · [机器可读前后数值](POST_COMPARISON.json)。三列视频依次为源 GT、checkpoint0、checkpoint128；每条都完整解码为 56 帧、24 FPS。

| 固定 probe | σ=0.25，MSE 0→128 | σ=0.75，MSE 0→128 |
|---|---:|---:|
| train constant | 0.233266→0.230171 | 0.071583→0.070220 |
| train switch | 0.253098→0.250999 | 0.091048→0.089797 |
| dev constant | 0.213012→0.210322 | 0.074761→0.073768 |
| dev switch | 0.215489→0.213123 | 0.081506→0.080181 |

8 个 probe 均小幅下降；均值 **0.154220→0.152322（-1.23%）**。这证明当前有限固定状态上对 FM 目标的拟合改善，不证明视频质量或动作响应改善。

| 视频 | Boundary MAD 0→128 | Frame MAD 0→128 | 关键观察 |
|---|---:|---:|---|
| train constant，GT history | 6.715→6.927 | 7.469→7.360 | 人物/地形延续，前后变化很小 |
| train switch，GT history | 10.718→10.866 | 8.577→8.350 | 人物未明显分解，动作收益不清楚 |
| dev constant，GT history | 16.087→16.415 | 17.560→17.414 | 人物完整，前后近似；较大边界差仍在 |
| dev switch，GT history | 8.424→8.395 | 7.889→7.826 | C2 轨迹与源视频有差异，前后近似 |
| dev constant，self history | 14.690→13.132 | 13.197→12.319 | 两者都偏离 GT 的运动路径；人物尚完整 |
| dev switch，self history | 10.124→7.134 | 5.920→5.621 | 边界数值下降，但前后轨迹仍近似，切换响应未获证实 |

直接播放入口：[train constant / GT history](paired_gallery/train_constant_gt_history_source_vs_0_vs_128.mp4)、[train switch / GT history](paired_gallery/train_switch_gt_history_source_vs_0_vs_128.mp4)、[dev constant / GT history](paired_gallery/dev_constant_gt_history_source_vs_0_vs_128.mp4)、[dev switch / GT history](paired_gallery/dev_switch_gt_history_source_vs_0_vs_128.mp4)、[dev constant / self history](paired_gallery/dev_constant_self_history_source_vs_0_vs_128.mp4)、[dev switch / self history](paired_gallery/dev_switch_self_history_source_vs_0_vs_128.mp4)。

MAD 是相邻帧差，并非视觉质量或动作正确性指标。六组三列视频已逐帧解码验证可播放；人工查看 0/18/38/39/43/55 帧 contact sheets，未见训练后出现新的大面积人物分解；GT-history 的 C2 人物/场景基本延续，self-history 也保有主体，但两条 dev 自生成轨迹与源 GT 明显分叉。checkpoint128 与 checkpoint0 的外观和运动差异有限，特别是 switch case 没有可确认的控制改进。代表帧无法排除间隔帧的短暂伪影，仍需 Judge 播放完整视频复核。这四条评测动作脚本没有从同一状态 fork 成 A/D 正反事实，因此不能给出 A/D 方向正确率或广义 action fidelity PASS。self-history 的前后比较也是 checkpoint 端到端能力对照，不可用来隔离 C2 adapter 的单一影响。

## 预算与故障账本

累计 [预算](CUMULATIVE_BUDGET.json)：**444/449 顶层 DiT、128/128 backward/update、70/70 VAE encode、14/14 VAE decode、151/600 text、6222.051/13800 GPU 秒**；训练单卡，未占用额外空闲卡。编码 1734.528 秒，训练成功尝试 2650.296 秒，post 660.890 秒。固定 pre 成功尝试 766.373 秒，三次失败分别为 301.064、81.937、26.962 秒，全部累计；评测/训练阶段限额未超。

运行环境为 `/home/qma/work/GWM/.venvs/h3world`，训练与评测使用空闲 GPU0，`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=4`。恢复训练入口为 `resume_run_v3.py --phase train_retry --gpu 0`，固定后评测为 `resume_run_v3.py --phase post --gpu 0`（1780秒 watchdog），离线展示片由 `finalize_cpu.py` 在 `CUDA_VISIBLE_DEVICES=-1` 下制作。冻结代码和输入 SHA 见 [训练来源清单](TRAIN_SOURCE_MANIFEST.json) 与 [恢复来源清单](TRAIN_RECOVERY_SOURCE_MANIFEST.json)；具体阶段用时和调用顺序见各原始日志/账本。

失败尝试的原账本仍为 `FAILED_STOPPED`，没有覆盖：首次 pre 缺少命令名 `ffmpeg`，13 DiT/1 decode；第二次 pre 在 BF16 身份记录 `.numpy()` 失败，9 DiT/1 decode；首次训练因审计 hook 安装早于 context 入口校验停止，保守记 2 DiT、0 backward/update。[attempt_00](attempt_00/README.md) · [attempt_01](attempt_01/README.md) · [attempt_02](attempt_02/README.md)。媒体修复与训练路由修复均有 CPU/真实门控证据；没有重跑已经完成的 pre 视频或源 RGB 编码。

## 结论与边界

EXP-023 验证了真实源 RGB 编码、严格因果 KV 下 C2 可微普通 FM、128-update checkpoint 恢复和前后匹配评测可以闭环。**模型能力仅是有限 probe 的小幅 FM MSE 改善；当前 56 帧视频没有显示明确的动作或画质提升，不能据此升级 V3 版本，也不能宣称 124 帧/跨场景/AnyFlow/DMD 已通过。** 这轮只训练 GT-history C2；C1 自生成分布和长时 rollout 没有进入训练目标。下一步是否扩数据、改变目标或研究 on-policy history，由 Judge 根据这些对照作独立决策；Worker 未启动衍生实验。
