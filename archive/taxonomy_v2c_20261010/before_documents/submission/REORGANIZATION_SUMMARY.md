# REORGANIZATION SUMMARY · 2026-10-10（V2a / V2b修订版）

本次已实际完成审计、版本映射、目录重组、CPU视频制作和汇报材料。按用户最新纠正，**V2a与V2b是同一生成历史/条件问题的并列修复路线，不是顺承版本**。未来V3才是高效严格因果、视觉与动作的统一目标。

## 1. 当前目录

```text
submission/
  mainline/
    V0_original_bidirectional/
    V1_native_chunk_causal/
    V2a_rgb_anchor_causal/
    V2b_same_sigma_local_bidir/
    V3_efficient_causal_planned/
  branches/
    A_causal_diagnostics/
    B_causal_adaptation/
    C_anyflow_dmd/
  report/
    README.md / index.html / roadmap.md / TALK_5MIN.md
    V0_original/ V1_native_causal/
    V2a_rgb_anchor/ V2b_same_sigma_local_bidir/
    00_comparison_gallery/       22条当前参考/对比视频
    01_research_branches_summary/
    02_next_steps/
  archive/
    historical_docs/ legacy_reports/ referenced_assets/
    reorganization_20261010/     首次整理审计与原编号收据
    taxonomy_v2a_v2b_20261010/   本次修订映射、制作与验收
    obsolete_presentation_v2_v3/ 旧编号MP4原字节保留
```

[汇报导航](report/README.md) · [浏览器本地演示](report/index.html) · [新路线图](report/roadmap.md) · [研究分支](branches/README.md)。

## 2. 移动、复制与归档

首次整理前，Git工作区干净；fetch后远端与本地基准`1a93713e218b8fd48d3fa5c47f25638068c012e0`一致，无需合并。初次整理归档4份历史文档并保留旧路径入口，复制71项核心来源、修复103处历史引用、补入27项旧资源。原始checkpoint、视频、log/CSV/JSON和生产代码不改写。

本次正式通过git mv将原V2/RGB目录改为V2a，原V3/Same-σ目录改为V2b，原V4 Planned改为V3 Planned；report目录同步。**18个带旧V2/V3屏幕标签的MP4连同副本归档保留**，没有覆盖。无版本字幕的原片副本仅改文件名，不重新编码。原实验outputs及checkpoint位置不变。

`experiments/`、`reports/`、`meeting/`继续作为原始证据层，mainline/branches通过相对链接与manifest组织；不搬坏冻结runtime。report为可以实际复制的汇报包，不含权重，无视频软链接依赖。

[本次逐文件映射与旧片哈希](archive/taxonomy_v2a_v2b_20261010/migration.json) · [初次复制清单](archive/reorganization_20261010/copy_manifest.json) · [历史断链修复](archive/reorganization_20261010/link_repairs.json)。

## 3. 实际视频交付

当前画廊共22条：保留6条V0/V1视频，新CPU编码16条V2a/V2b标签视频；这些新片另有7份版本目录实体副本。所有内容来自已有真实MP4，H.264/yuv420p、24fps、等比例留边、逐帧编号，无补帧、循环、慢放或模型推理。

| 主比较 | 帧数 | 目的 |
|---|---:|---|
| [V0 vs V1](report/00_comparison_gallery/V1_vs_original.mp4) | 124 | 工程因果化的动作/结构代价 |
| [V0 vs V2a](report/00_comparison_gallery/V2a_vs_original.mp4) | 124 | RGB联合视觉修复的结果与动作局限 |
| [V1 vs V2a](report/00_comparison_gallery/V2a_vs_V1.mp4) | 124 | 零训练latent-dual与RGB联合方案，非anchor单项消融 |
| [V0 vs V2b](report/00_comparison_gallery/V2b_vs_original.mp4) | 56 | Original参考前56f vs局部Same-σ路线 |
| [V1 vs V2b](report/00_comparison_gallery/V2b_vs_V1.mp4) | 56 | 恢复原生条件/局部双向重算的并列修复路线 |
| [V2a vs V2b](report/00_comparison_gallery/V2a_vs_V2b.mp4) | 56 | 并列能力比较；8/30步、clean KV/Same-σ重算等不同 |
| [V2b四路径](report/00_comparison_gallery/V2b_four_paths_56.mp4) | 56 | AA/AD/DA/DD，自身history，RGB39边界，无GT reset |

每组均有逐A/逐D片和合并片。[完整画廊](report/00_comparison_gallery/README.md)。V2a完整124f原片和20秒失败片、V2b四条完整56f原片均保留；裁短比较不替代完整失败证据。

视频内清晰标出当前版本、采样步数、history/KV差异；V2a vs V2b顶部明确CAPABILITY COMPARISON / NOT a single-variable ablation。计时仍是原运行记录：124f全片E2E或V2b第二块sampling，不伪称相同测量scope。

## 4. 最终版本结论

| 版本 | 已验证什么 | 未解决什么 |
|---|---|---|
| V0 | Original动作与画面正控 | 非GT、非长期稳定保证 |
| V1 | strict causal与persistent KV工程执行 | 本代表动作近停滞、后段过亮/退化；联合能力未过 |
| V2a | Longer-horizon Visual Stability Demonstrated：**仅124f人物/场景相对稳定证据** | A/D方向失败；20s崩坏，未证明端到端加速 |
| V2b | Local Visual and Action Fidelity Demonstrated：**仅56f第二块四路径局部正控** | 不支持persistent hidden KV；无第三/第四块、124f与长时效率验收 |
| V3 | **Planned Unification**，目标是严格因果+历史复用+视觉+动作 | 无模型/视频；不能直接拼接V2a/V2b checkpoint宣布成功 |

V2a是RGB image-conditioning、visual QKV、endpoint/boundary与routing等联合协议，不能单归因anchor。V2b从Original恢复Single I0/native time，无V2a visual adapter；Same-σ配合T2局部双向重算，不是strict chunk-causal masked attention。

## 5. 缺失材料

- MISSING_MATCHED_VIDEO：RGB39切换的Original/V2a A→D、D→A完整56f对照；现有WAD/DSA的切块协议不同，不能冒充。
- V2b多块/跨场景/124f，V3严格因果骨干、缓存与动作/结构/效率联合验收。
- 基于可信统一骨干的AnyFlow和完整on-policy DMD结果。旧AnyFlow16/64/128/136及DMD-lite是旁路探索，不是V2b已完成Stage1/2。
- V0/V1早期逐次运行的全部冻结runtime；包内有可追溯代码快照但不冒充当时逐字节源码。V2b冻结runner/runtime哈希有核验。
- V1同口径MAD/boundary、统一硬件warmup多次均值、权重/激活/KV分项测量。真实ABot主要FM/E2，没有找到真实ABot AnyFlow训练结果。
- 非实验材料：上游ABot展示海报未找到，已明确说明，未留下坏链接。

## 6. 不可作严格消融的比较

V2a/V2b分别改变checkpoint来源、anchor、时间/精度、history、attention/cache、chunk和steps。它们解决同一研究问题的不同侧面，没有顺承关系；比较是能力取舍，不能把收益归因单因素。V1→V2a也改变多个因素。124f裁56f和真实12+5窗口的可见未来不同；39f/124f/17新帧flow不可直接混算恢复比例。

8steps/chunk×8=64noisy+8commits，与Original30整段calls不能按30/8算加速。cache/recompute数值等价不代表Original动作/画质等价，cosine和MAD不单独作为质量结论。

## 7. 完整性与验收

最终验收 **PASS**：修订前3141个文件全部可追踪，其中3094个保持原字节（含改名/归档），47个仅为现行说明或导航更新；1059个重点证据文件通过保护检查。79份打包MP4（62个唯一内容，含旧标签归档）全部解码通过，当前22条画廊帧数与配方一致；338份Markdown/HTML中的1938处本地链接无断链。V2b冻结runtime哈希、71项核心复制和15项源码快照一致，22份Python通过语法检查。report含94个文件，117581957 bytes（约112.1MiB），实际独立复制与无软链接依赖检查通过。

[本次修订验收](archive/taxonomy_v2a_v2b_20261010/validation.json)核对迁移前后每个旧MP4、权重、代码和测量；检查当前视频完整解码/FPS/PTS、当前链接、源码hash及report隔离复制。旧编号编码和初次审计收据保留，可通过migration追踪，不改写原始实验结论。

[初次整理验收](archive/reorganization_20261010/validation.json)描述改名前的历史目录；其路径须按本次migration定位。没有训练、GPU模型推理、AnyFlow或DMD。本次整理内容纳入Git提交并发布至仓库；实际发布版本以Git提交及远端分支为准。审计收据中的commit_created/pushed字段记录的是发布前检查时的状态。
