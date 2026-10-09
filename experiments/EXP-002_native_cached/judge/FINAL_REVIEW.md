# EXP-002 / v1 — Judge final review

2026-10-10，Asia/Hong_Kong。**Decision: accept；任务关闭。接受73帧严格缓存候选的可行性证据，正式124帧V3尚未完成。**

## 结论与范围

| 维度 | 判定 | 证据与边界 |
| --- | --- | --- |
| 执行与协议 | PASS | Original H3 + released action LoRA，native Single I0，12→5→5，30步/chunk；0训练、无GT重置、未来video/action物理裁剪 |
| Strict chunk causal / persistent KV | PASS（本协议） | 源码与CPU检查、首12模型身份relative RMS/maxabs=0、真实50层raw video KV持久化与自己的第二块commit；后续采样不重算历史 |
| 动作可行性 | PASS（单scene/seed、73帧） | 同first12_A下第二块AA/AD flow +1.347/−1.458且视觉响应可辨；第三块+0.474/−0.930为各自历史续写，不能当同状态反事实 |
| 人物/场景可行性 | PASS，有质量限制 | 新增RGB39–72全部静态序列与原尺寸端点已看；单体人物、停车场可辨；AA55→56明显姿态位置跳变、节奏与细节软化保留 |
| 效率记录 | PASS（增量成本）；完整E2E/公平speedup NOT_TESTED | 124 forwards、4 VAE、825.238 GPU秒；缓存真实复用不等于Original公平加速 |
| 124帧与跨场景 | NOT_TESTED | 下一步只延伸同一候选，不追加短片消融或训练 |

共享首12 latent是既有模型生成历史，首39 RGB复用既有生成片，非GT。候选空cache图的真实模型身份诊断支持首窗复用；73帧不能描述为本轮全部从零采样。源模型、输入与代码凭据见[MANIFEST](../MANIFEST.md)、[Judge源码摘要](source_manifest.json)和逐块原始JSON。

## 审阅内容与可追溯证据

Judge独立读取interval入口、两版runner、路由/缓存代码与测试。9项本轮CPU输入/位置检查和10项既有current-prefix检查通过；不重复18状态模型诊断。独立检查两条第二块输入配对、第三块恢复cache链、endpoint及published RGB摘要；每条已显示56帧保持冻结。完整解码56/73原片；新增全部帧的静态图与原尺寸端点见本目录，不声称原速播放。最终20项交付manifest均匹配，两条73帧并排片完整解码、24fps/73帧/PTS通过，布局抽看通过。

- [阶段审阅](S0_REVIEW.md)、[同history配对](second_pair_checks.json)、[AA第三块](AA_third_checks.json)、[AD第三块](AD_third_checks.json)、[交付检查](delivery_checks.json)。
- [冻结Worker原始报告](../worker_report.md)、[原始metrics快照](../worker_metrics_snapshot.json)、[执行任务书](../accepted_taskbook.md)。原始报告中的相对链接按根report.md位置写，实验入口提供当前有效导航。

## 成本与限制

120 sampling +3 prefill/commit +1首窗身份诊断 =124 forwards；4 VAE、0 optimizer、单GPU0依次执行，0.22923284 GPU-hours，首至末914.45秒。最大已记录allocated26,686.14MiB。首12 cache6,799,104,000 bytes；through17为9,632,064,000 bytes，50层各6630历史video tokens。

首条AA在instrumentation增强前执行：仅磁盘cache hash不变，没有进程内identity/version记录；peak在prefill后reset，缺该阶段峰值。冻结原代码并披露，不重跑。后三次进程有内存只读与各阶段峰值；这些证据结合源码足以支持本轮可行性判断。原JSON的commit_calls_per_layer实际为累计层级提交数，不能当模型forward次数。

每次增量进程包含重新加载/缓存序列化，不能当完整首屏或73帧从零E2E；V2b同时改变拓扑、history sigma及缓存，不可单因素归因。普通画质缺陷按用户可行性尺度接受；单停车场seed13、仅AA/AD、73帧不覆盖泛化和长期质量。

## 下一步决策

方向已有有效信号，下一项有价值的验证是同配置自己的历史续写73→124，记录逐块缓存与实际成本。不给新场景、seed、训练或额外诊断预算。若出现严重持续失效则记录边界、停止相应方向，不为挽救结果无限追加实验。EXP-003须在本轮归档与Git交付后正式发布。
