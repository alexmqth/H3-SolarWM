# V3 Baseline：早期73帧证据

## 名称与目的

**名称解释：** 73是输出RGB帧数；AA为持续A，AD为从A切换D；KV为历史注意力键和值缓存。

**本次目的：** 记录正式124帧验收之前的原生条件与缓存可行性证据，帮助追溯基线来源。

[其他缩写与计数](../../../GLOSSARY.md) · [汇报总览](../../../README.md)

## 实验结果与证据

EXP-002/v1已获Judge验收，本页保留该阶段73帧证据；后续EXP-003已完成124帧验收，见[分支主页](../README.md)。Original H3 + released action LoRA；native Single I0、12→5→5、30steps/chunk、private current-prefix、sigma0提交的persistent raw video KV，零新增训练。

- [左V2b、右candidate AA73](videos/V2b_vs_V3_Baseline_AA_73.mp4)
- [左V2b、右candidate AD73](videos/V2b_vs_V3_Baseline_AD_73.mp4)
- [原始证据与配置](../../../../experiments/EXP-002_native_cached/README.md) · [Judge验收](../../../../experiments/EXP-002_native_cached/judge/FINAL_REVIEW.md)

同A首窗下第二块A/D响应可辨，flow +1.347/−1.458；第三块各接自己的历史，+0.474/−0.930。人物与停车场基本可用，AA55→56有明显姿态跳变，画质/节奏有缺陷。所有新增帧经静态序列审阅；不把静态审阅称为原速播放。

124次新增forward、4VAE、0.22923GPU-hours，已记录peak26,686MiB；首条AA缺prefill峰值及进程内cache不变记录，后三次补齐。首73并非从零生成的E2E计时，不给公平Original加速比。两侧同时改变拓扑/history sigma/cache，非单变量消融；首39后生成历史不同。Original A→D匹配基线缺失。

该阶段后续扩展已由EXP-003完成；当前下一决策见[研究建议](../../../02_next_steps/README.md)。单scene/seed可行性不等于成熟画质或泛化。
