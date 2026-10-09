# H3-World × SolarWM：因果化、动作信息流与少步探索

**研究目标：** 将SolarWM的causal chunk、KV cache与少步方法迁入H3-World，检查长视频效率、画面连续性和action control能否同时保留。

**当前判断：EXP-002的73帧严格缓存候选可行性已验收；正式V3待同配置124帧。** 同history A/D响应、真实KV复用和基本人物/场景结构同时得到有限证据；AA跨块跳变及单scene/seed限制保留。0训练、124新增forward、0.22923GPU-hours；未测完整E2E或公平加速。[最新候选](report/V3_native_cached_candidate/README.md)。V2b持续A/D124帧与Original对比保留为已验收参考。

## 一分钟入口

| 使用场景 | 入口 |
|---|---|
| 组会直接展示 | **[report/README.md](report/README.md)** / [浏览器本地演示页](report/index.html) |
| 连续播放全部对比 | [比较画廊](report/00_comparison_gallery/README.md) |
| 当前局部正结果 | [V2b四路径56f](report/00_comparison_gallery/V2b_four_paths_56.mp4) |
| 核心模型/协议演进 | [mainline V0–V3](mainline/README.md) |
| 机制、训练、AnyFlow/DMD细节 | [Research Branches A/B/C](branches/README.md) |
| 面试题回答与报告 | [INTERVIEW_ANSWER](INTERVIEW_ANSWER.md) / [REPORT](REPORT.md) |
| 运行环境、权重、adapter | [REPRODUCE](REPRODUCE.md) / [checkpoints](checkpoints/README.md) |
| 本次移动/复制/视频/验收 | [REORGANIZATION_SUMMARY](REORGANIZATION_SUMMARY.md) |

## 版本比较

| 版本 | Action fidelity | Visual stability | 已验证范围 | Persistent KV | 采样 | 新增训练 | 当前结论 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D方向正控 | 124f基本完整 | 124f及已有长片参考 | 否 | 30整段；另存50步 | 无，released LoRA | Reference，非GT |
| V1 Native causal | 显著下降 | 本代表停滞、过亮/背景退化；其他早期协议有重影 | 124f工程rollout | 是，CPU raw video KV | 8/chunk | 本代表无 | 工程可行，联合质量失败 |
| V2a RGB-Anchor | A/D方向失败 | **124f人物结构相对稳定**；20s失败 | 124f视觉证据；243/481f负结果 | 是，clean commit的历史hidden KV | 8/chunk | visual adapter +已训action residual | Longer-horizon Visual Stability Demonstrated（仅124f scope） |
| V2b Same-σ local bidir | 持续A/D在124f有可辨响应；切换仍有限制 | 124f人物/场景基本可用；边界与节奏有缺陷 | 单停车场、seed13、六块124f | **否**，每步重算全部可见历史 | 30/chunk | **无**，Original + released LoRA | Sustained A/D feasibility accepted；非V3 |
| V3 Efficient causal | 候选同history A/D响应可辨 | 候选73f基本可用，AA边界跳变 | EXP-002 AA/AD73f已验收；正式124f待验证 | 是，strict chunk causal + raw KV | 30/chunk | 无，Original + released LoRA | 73f feasibility candidate；完整V3 pending |

V2a与V2b共同研究生成历史/条件不匹配的修复，没有顺承关系。V2a为RGB联合视觉适配；V2b从Original恢复Single I0/native条件，Same-σ与局部双向重算，已扩展至持续A/D的124帧，切换仅有局部证据。未来V3才是严格因果、KV、视觉和动作的统一目标。

## 目录

```text
mainline/             V0 → V1 → {V2a RGB, V2b Same-σ} → V3 Planned
branches/             A机制诊断 / B因果适配与动作恢复 / C AnyFlow与DMD探索
report/               可独立复制的简洁汇报：核心源码、真实视频、版本说明、5分钟讲稿
archive/              历史文档、旧导航、整理前索引、hash与视频制作/验收收据
code/ checkpoints/    现有运行实现与小adapter；保持原路径
experiments/ reports/  原始实验说明、冻结源码、指标/日志/视频；按新分支索引，原路径保留
meeting/              原会议包与旧主片；作为历史证据保留，最新展示使用report/
docs/ scripts/ tests/  原长文档、运行脚本与测试；未修改生产模型实现
```

旧`reports/stage1_anyflow`是历史文件夹名，含FM、诊断等多个目标，不等于全是AnyFlow。旧结果与measurement不搬乱；新分支逐项建立链接和manifest，减少对原运行脚本的路径破坏。原始大checkpoint和latent仍在外部outputs，由来源清单关联。

**计数口径：** 8steps/chunk×8chunks=64 noisy forwards + 8 commits，不是全视频8次。KV复用不等于AnyFlow少步训练，DMD-lite不等于完整Stage2。耗时为共享硬件单次记录，无warmup均值或公平speedup结论；flow/cosine/MAD均不能单独代替动作与画质评审。

底座与released LoRA不在仓库内，见复现文档。EXP-001新增V2b受控推理，0训练；此前整理记录保持历史口径。
