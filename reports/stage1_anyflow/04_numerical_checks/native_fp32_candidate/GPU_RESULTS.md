# 原生FP32候选：真实GPU训练与效果记录

快照：2026-10-08T06:47:38.173130+08:00。GPU0独占，reserve6，CPU raw KV。

这轮只修正精度协议：原生F32输入/输出/时间权重、FP32时间混合/SiLU与有限差分输入，主block仍BF16。保持同一visual/action初始化、RGB dual、chunk5、history5、seed13、shift2.22和tail16 QKV rank8；time MLP冻结。

第一次真实更新已完成：QKV变化、time严格不变；train loss=0.09402979，QKV梯度范数0.00443731，update=60.21秒，含准备/前后验证308.71秒，allocated峰值38556.35MiB。A held-out endpoint raw59.625→74.741、D22.805→23.891，不能称为改善。

step01→16恢复时adapter、Adam、logical/CPU/CUDA RNG、历史记录及teacher身份逐项一致；见[native_gpu_resume_audit.json](native_gpu_resume_audit.json)。这验证加载状态，不声称与未中断CUDA训练轨迹逐bit相同。

| 初始化step00 | Horizontal flow | E2E s | GPU allocated peak MiB | CPU KV MiB | Noisy + commit |
|---|---:|---:|---:|---:|---:|
| A / 4 steps/chunk | -1.233794 | 188.2 | 39069.2 | 6484.1 | 12 + 3 |
| D / 4 steps/chunk | -1.225326 | 190.9 | 39069.2 | 6484.1 | 12 + 3 |

A-D=-0.008468；A错误、动作gate失败。两条视频完整解码39帧，首帧条件/prompt/action rows/video/audio noise与原始teacher逐张量相等。抽帧0/10/20/30/38显示约20帧开始明显ghosting，后段人物/场景严重雾化，与legacy初始化没有明确视觉修复。见[native00_vs_original_AD4.jpg](native00_vs_original_AD4.jpg)。这项视觉结论限于该初始化对照，不提前判断训练后的结果。

总16次训练已完成。step01→16这次运行含准备/前后验证1436.1秒，allocated峰值38652.9MiB；QKV更新、time保持冻结。这不是累计1–16的训练总耗时。

| 同noise验证 | step01 | step16 |
|---|---:|---:|
| A weighted total | 0.117999 | 0.117049 |
| A endpoint raw | 74.740799 | 55.934113 |
| A flow_map raw | 0.168687 | 0.167230 |
| D weighted total | 0.170234 | 0.169150 |
| D endpoint raw | 23.890673 | 22.690615 |
| D flow_map raw | 0.225160 | 0.224058 |

A endpoint raw从74.741降到55.934，D23.891降到22.691；相对训练前step00的59.625/22.805仍只有有限变化。它们是固定noise的数值验证，不能当作画质/动作收益。训练后的A/D4/8-step正在生成，质量结论待实际视频。

数值候选已合入主源码/提交包为显式可选profile，legacy默认保留；运行中runtime和manifest不变。主源码完整回归41 passed in5.90s；从记录的DiffSynth base+导出patches重建，模型/pipeline与测试源码相同，17项集成检查及6项官方oracle检查通过（oracle补充SolarWM/src路径后运行）。见[native_export_reproduction.json](native_export_reproduction.json)。

实现和恢复检查通过不代表Stage1画质验收通过；Stage2继续暂缓。


训练后第一条A/native4已完成：39帧完整解码，输入conditioning逐张量公平；水平flow=-1.251782，A方向仍错误。E2E226.71秒、allocated peak39069.22MiB、CPU KV6484.13MiB、12 noisy forwards+3 commits；Gray MAD4.2868、Boundary RGB MAD6.4753。抽帧0/10/20/30/38显示后段严重ghosting/雾化，与step00没有明确修复；同初始化FM16的4-step画面明显更完整（其精度为legacy，不能作仅loss变量的严格对照）。见[native16_A4_vs_initial_FM_teacher.jpg](native16_A4_vs_initial_FM_teacher.jpg)。该4-step设置已经因A方向和抽帧退化失败；D4、A/D8仍在生成，暂不推断它们的结果。
