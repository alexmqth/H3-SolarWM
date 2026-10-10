# EXP-007/v3 AF3 — Worker 视频评估报告

2026-10-11 HKT。状态：**已执行完毕，等待 Judge 能力验收**。本轮只评估 AF2 step32 checkpoint；没有新增训练、checkpoint 扫描、NFE 扫描或 DMD GPU 运行。

## 问题与冻结协议

验证有限 AnyFlow 训练后的 target-time student，能否在 V3 strict causal、persistent CPU raw KV 下，以 8 finite-map forwards/chunk 保留人物、停车场和 A/D 方向，并与普通 FM8 对照。使用停车场 seed13、native shift2.22 九 sigma 点、12+5+5 latent（73 RGB 帧）、Single I0、current-prefix feedback、Global RoPE、h3_fp32 边界。AF2 step32 的 target-time gate0.25 与 last8 rank8 QKV 同时生效。首 39 RGB 帧与 EXP-006 FM8 **逐值相同**；AF student 以自身权重建立该首块 KV。C2 是共同历史上的比较，C3 已各自使用自己生成的 C2 历史。AF2 训练 C1 使用原 V3 30-step 历史，而 AF3 评估 C1 使用 FM8 历史，属于分布差异。

来源冻结在 [source_manifest_v3_af3.json](source_manifest_v3_af3.json)，执行配置在 [config_v3_af3.json](config_v3_af3.json)，runner 为 [run_af3.py](run_af3.py)。AF2 配套 checkpoint 位于 `H3-World/outputs/EXP-007_v3_anyflow_af2/step_32/`，未复制33B基础权重或训练状态到提交包。

## 执行与资源

GPU0，`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`，`.venvs/h3world/bin/python -u submission/experiments/EXP-007_v3_anyflow/run_af3.py --stage STAGE --path PATH --gpu 0`。批准顺序：shared `prefill`，`second AA`，`second AD`，`third AA`，`third AD`。各次 stdout 原件位于 `H3-World/outputs/EXP-007_af3_*`，逐调用账本原件位于 `H3-World/outputs/EXP-007_v3_anyflow_af3/budget.json`。

第一次 prefill 因遗漏 `benchmark` 导入路径，在加载模型与任何 forward 前失败：0 forward、0 VAE、3.561 秒。保留 [事故记录](AF3_ATTEMPT1_FAILURE.md) 与首次 source manifest。Judge 批准修复后一次重跑；累计预算未重置。随后五个阶段均完成。总计 **35 forward = 32 sampling + 3 clean commit，4 VAE，0 optimizer update，473.000 GPU 秒（0.1314 GPU 小时）**；低于0.35 GPU小时，最终产物各73帧、24 FPS、832×480，可用 PyAV 完整解码。单段 allocated 峰值最高约26.34 GiB；C1 CPU raw KV 6.33 GiB，C2 commit 后约8.97 GiB。没有占用 GPU3/4 上的他人进程。

| 路径/块 | AF8 水平光流 px | FM8 对照 px | AF8 边界/块内灰度 MAD | FM8 边界/块内灰度 MAD | AF8 采样秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AA C2, 39–56f | +0.650 | +0.846 | 4.316 / 3.740 | 3.658 / 4.147 | 40.573 |
| AD C2, 39–56f | −0.697 | −0.321 | 4.408 / 4.307 | 3.851 / 3.932 | 38.577 |
| AA C3, 56–73f | +1.006 | +0.505 | 6.060 / 3.986 | 4.451 / 3.592 | 43.762 |
| AD C3, 56–73f | −0.149 | −0.826 | 11.275 / 2.527 | 4.568 / 3.826 | 41.999 |

水平光流是镜头运动代理，不是严格动作准确率。C2 A/D 有相反符号；C3 AD 仍是负号但幅度减弱。AF8 各段人物和停车场仍可辨，姿态与透视有跳变；AD C3 的边界尖峰尤其明显。关键帧及全部新增帧人工检查后，**有限生成可行、质量 PARTIAL；目前没有证据表明 AF8 整体优于普通 FM8**。AF 与 FM8 同为8 NFE，AF 还多了训练，因此本轮不能宣称端到端加速或纯目标函数单因素收益。此单场景/单 seed/73 帧结果不支持124帧或跨场景泛化结论。

## 可播放产物与追溯

| 路径 | 原片 | 普通 FM8 与 AF8 并排 | 关键帧 |
| --- | --- | --- | --- |
| AA | [73帧原片](artifacts/af3_raw/AA/rollout_73.mp4) | [73帧并排](artifacts/af3_videos/AA_FM8_vs_AF8_continuation_73.mp4) | [关键帧](artifacts/af3_videos/AA_keyframes.jpg) |
| AD | [73帧原片](artifacts/af3_raw/AD/rollout_73.mp4) | [73帧并排](artifacts/af3_videos/AD_FM8_vs_AF8_continuation_73.mp4) | [关键帧](artifacts/af3_videos/AD_keyframes.jpg) |

每块 JSON、prefill 结果及最终 budget 的小文件副本在 [artifacts/af3_raw/](artifacts/af3_raw/)。两条并排视频均为 H.264、73帧、24 FPS、1248×424；[manifest](af3_comparison_manifest.json)记录两侧 SHA 和协议差异。所有原始视频及完整 latent/KV 仍在 `H3-World/outputs/EXP-007_v3_anyflow_af3/`；提交包中的副本不覆盖原件。

## Worker 判断

AF2 训练与 AF3 视频证明 target-time student 可以走完这一有限8-map causal/KV 续写流程，A/D 局部响应未塌陷，人物结构也没有持续崩坏。与普通 FM8 相比没有稳定的画质或边界收益，AD 第三块边界明显更差。是否接受为有限 AnyFlow feasibility、是否进入独立 DMD pilot，由 Judge 在复核视频和原始账本后决定；本报告不升级正式 V3 版本。
