# V3：因果缓存基线、Sliding Window 与 FM → AnyFlow

V3 的共同基础是 **Strict Causal（严格因果）+ Persistent KV（持久历史键值缓存）+ 原生 H3 动作条件**。正式参考仍为已验收的 30 步、124 帧 V3 Baseline。按研究问题查找下面两条路线；DMD 已归入 [V4：SGF / DMD](../v4/README.md)。

| 入口 | 要解决的问题 | 已有结论 | 视频位置 |
| --- | --- | --- | --- |
| [V3 Baseline](v3_baseline/README.md) | 因果生成能否同时保留动作、人物结构与历史 KV 复用？ | 124 帧有限可行；画质与连续性 PARTIAL | `v3_baseline/videos/` |
| [Sliding Window](sliding_windows/README.md) | 长视频的历史 KV 能否固定容量？Global / Local 位置如何影响生成？ | Global 158 帧有限可行；当前无训练 Local 没有联合收益 | `sliding_windows/global/videos/`、`local/videos/` |
| [FM → AnyFlow](fm2anyflow/README.md) | 能否降低每块采样成本？直接减步与专门少步训练分别有什么收益？ | FM-8NFE 有限可用；AF-8NFE 未显示整体替代收益，AF-4NFE 当前配置失败 | 各尝试的 `videos/` |

```text
v3/
├── v3_baseline/                 # 正式 30 步 / 124 帧基线
├── sliding_windows/
│   ├── global/                  # V3-SW-G：全局位置
│   └── local/                   # V3-SW-L：滑窗局部位置
└── fm2anyflow/
    ├── fm8_continuation/        # EXP-004：30 步首窗后的 8 步续写
    ├── fm8/                     # EXP-006/010：全程 8 步及 158 帧滑窗
    ├── fm4_vs_af4/              # EXP-009：普通 4 步与 AnyFlow 4 次求值
    ├── anyflow/                 # EXP-007/012：AF-8NFE 训练与迁移
    ├── scene_transfer/          # EXP-011：FM-30NFE / FM-8NFE 工业、村落对照
    └── data_preparation/
        ├── teacher_generated_targets/  # EXP-014：模型生成的候选监督视频
        └── real_data/                  # EXP-015/016/017：录屏、VAE、CPU 布局
```

先读各页开头的“名称与目的”，再看 `videos/`，最后按需打开实验及 Judge 审核。分组目录负责导航；有生成或重建结果的叶子目录存放实物 MP4。数据准备页面明确区分模型生成、真实录屏和重建视频；CPU 布局测试没有生成视频。

[常用缩写与计数](../GLOSSARY.md) · [完整汇报导航](../README.md) · [浏览器视频播放器](../index.html)
