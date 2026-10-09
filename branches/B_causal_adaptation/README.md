# B_causal_adaptation

按研究问题归档；下表区分实现、训练完成和能力验收。原报告/原始日志保留原路径与字节。

| 子实验 | 状态 | 结论 |
|---|---|---|
| [Tail-QKV ordinary causal adaptation](01_tail_qkv/README.md) | preliminary trained; joint quality failed | 普通FM causal adaptation已训练，并不等于AnyFlow；加载训练容器也不代表optimizer已经更新。 |
| [Fixed-mix / scheduled sampling](02_history_mixing/README.md) | preliminary trained; visual/action trade-off | 旧fixed-mix124f A/D=+0.1427/−0.3105，比V2a符号好但尾段撕裂；不是Original级控制，也不是零训练native V1。 |
| [Online per-sigma teacher replay](03_online_replay/README.md) | preliminary trained; quality failed | 逐sigma replay及paired训练链路存在；内部loss/cosine改善没有可靠转化为自由rollout方向与视觉联合PASS。 |
| [Action velocity / score-delta / endpoint](04_action_geometry/README.md) | diagnosed; recovery failed | 区分同一history/noisy state的反事实差分与不同轨迹相减。多种action路径适配未过，不能用差分cosine独自认定视频好。 |
| [RGB visual + Original endpoint supervision](05_rgb_visual/README.md) | preliminary trained; 124f visual improved, action failed | 对应V2a联合协议；普通replay/endpoint两更新，不是Stage2。Stage2-lite后来集成是另一个实验。 |
| [Real ABot FM48 / noise-density FM48](06_real_abot_fm/README.md) | preliminary trained; quality failed | 真实录制ABot：16训练/8验证，按episode拆分，48更新；objective为ordinary FM。GT history与generated history分别验收，后者仍分解。 |
| [FM-only vs FM+Action consequence ranking](07_e2_action_ranking/README.md) | preliminary trained; no consistent gain; frozen | A-history、D-history和两条真实GT-history全收齐；两臂各4更新，无一致额外收益。没有自动扩到16。 |

[返回研究分支总览](../README.md) · [主线版本](../../mainline/README.md) · [精简汇报](../../report/README.md)
