# H3-World × SolarWM：因果化、动作信息流与少步探索

**研究目标：** 将SolarWM的causal chunk、KV cache与少步方法迁入H3-World，检查长视频效率、画面连续性和action control能否同时保留。

**当前判断：工程迁移成立，完整能力目标尚未完成。** strict causal/KV能生成124帧；RGB联合修复改善124帧结构但A/D失败、20秒崩坏；最新V3从Original重新出发，Same-σ + C12→5 + T2联合重算在自身history后的第二块取得局部动作与结构正结果。V3没有persistent hidden KV，不代表完整124帧、AnyFlow或Stage2已经成功。

## 一分钟入口

| 使用场景 | 入口 |
|---|---|
| 组会直接展示 | **[report/README.md](report/README.md)** / [浏览器本地演示页](report/index.html) |
| 连续播放全部对比 | [19条画廊](report/00_comparison_gallery/README.md) |
| 当前局部正结果 | [V3四路径56f](report/00_comparison_gallery/V3_four_paths_56.mp4) |
| 核心模型/协议演进 | [mainline V0–V4](mainline/README.md) |
| 机制、训练、AnyFlow/DMD细节 | [Research Branches A/B/C](branches/README.md) |
| 面试题回答与报告 | [INTERVIEW_ANSWER](INTERVIEW_ANSWER.md) / [REPORT](REPORT.md) |
| 运行环境、权重、adapter | [REPRODUCE](REPRODUCE.md) / [checkpoints](checkpoints/README.md) |
| 本次移动/复制/视频/验收 | [REORGANIZATION_SUMMARY](REORGANIZATION_SUMMARY.md) |

## 版本比较

| 版本 | Action fidelity | Visual stability | 长时 rollout | Persistent KV | Sampling steps | 新增训练 | 验收状态 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D 正控；W/S 以视频为准 | 124f 基本完整 | 有243/481f参考；仍有几何变形 | 否 | 30整段；另存历史50步 | 无，released LoRA | Reference，不是GT |
| V1 Native causal | A/D显著减弱；本代表A符号错 | 停滞、透明/重影等退化 | 124f执行完成 | 是，CPU raw video KV | 8/chunk × 8 + 8 commits | 本代表无 | 工程可行，质量/动作未过 |
| V2 RGB联合修复 | A符号错；未恢复 | 124f相对改善；20s失败 | 124/243/481f均有片 | 是，CPU raw video KV | 8/chunk | visual QKV + action residual | 视觉局部改善，联合验收失败 |
| V3 Same-σ C12→5 | 两份自身history下当前A/D方向正确 | 第二块人物结构保持 | 仅56f、2块 | **否**，T2逐sigma联合重算 | 30/chunk | 无，从Original重新出发 | **Current Best Local Causal Rollout Candidate** |
| V4 Efficient causal | 待验证 | 待验证 | 待验证 | 目标：是 | 先30/chunk可信，再少步 | 待定 | Planned，无checkpoint/视频 |

V1主片明确选**零新增adapter、latent dual、seed13、8steps/chunk**的native配置；最早seed2/4-step另存，fixed-mix训练归Branch B。V2视觉结果来自RGB anchor + visual adapter +监督等联合协议，不能只归因anchor。V3从Original恢复原生条件，不继承V2 checkpoint；其视频比较是跨协议研究对比。

## 目录

```text
mainline/             V0 Original → V1 native → V2 RGB → V3 Same-σ → V4 Planned
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

底座与released LoRA不在仓库内，见复现文档。本次只整理现有证据并CPU编码对比，无训练、33B推理、AnyFlow或DMD运行。
