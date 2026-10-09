# FM32 对照已全部完成

最新完整结论、指标与视频见 [FINAL_RESULTS.md](FINAL_RESULTS.md)。以下保留训练完成时的审计记录；其中“等待评测”是历史状态。

# 匹配full-history/shift12/FM32训练已完成

记录时间：2026-10-08T11:44:58.793174+08:00。含准备/验证3215.65秒，allocated峰值41251.94MiB。A/D4/8视频评测已开始，尚不能据此排名画质。

真实GPU起点三套adapter、CPU/CUDA/logical RNG与AnyFlow一致；最终32次action/chunk/current sigma与AnyFlow实际日志逐项相同。两者全覆盖rank8、43,237,376可训练参数、同数据/训练shift12/validation2.22/seed等共有配置一致。FM始终r=t，不能要求它匹配AnyFlow的off-diagonal目标r。

| 32次更新的训练前向计数 | FM | AnyFlow |
|---|---:|---:|
| loss样本 | 128 | 128 |
| 当前chunk model evaluations | 128 | 512 |
| no-grad clean commits | 30 | 30 |
| 带梯度历史重建 forwards | 120 | 120 |

这些是不含验证和反向checkpoint重算的记录计数；相同更新数并不等计算预算。AnyFlow分两段启动/验证，FM连续32，所以脚本wall time不能作严格吞吐比。预测noise hash没有和actual GPU noise hash比较。

| Action | 相同diagonal sample | FM raw loss | AnyFlow raw loss |
|---|---:|---:|---:|
| A | 1 | 0.064127550 | 0.064365514 |
| A | 2 | 0.140702829 | 0.140805602 |
| D | 1 | 0.096416526 | 0.096661933 |
| D | 2 | 0.198561966 | 0.198720217 |

这里只比较各动作前两个公共r=t样本。FM raw residual略低，但只有这几个验证点，不能由此断言视频效果或统计显著性；两种目标的total loss不可直接排名。实际视频仍待完成及逐帧检查。

[完整实际日志审计](matched_training32_audit.json)、[初始张量/RNG及全sigma序列检查](gpu_matched_initialization.json)。Stage1效果尚未验收，Stage2暂缓。

## 首条A4完整视觉检查

FM32的A/39f/4 steps per chunk已完成，flow=−1.058372，A方向仍错。全部0–38帧contact sheet与Original/AnyFlow32/FM32在12/24/30/38帧的对应画面已审阅：FM仍有模糊、透明感，但人物与车库明显比AnyFlow32/4-step完整，后者约20帧后严重雾化。这是静态完整帧复核，不声称实时播放；结论限于首条A4，其余D4/A8/D8尚待完成。

[同帧对照图](fm_shift12_32_A_4step_comparison.jpg)、[完整39帧contact sheet](fm_shift12_32_A_4step_all39.jpg)、[完整FM A4原视频（无展示标注）](FM32_A_4step_raw.mp4)。相同初始化、参数容量、数据/sigma序列和32次更新；同更新数不等计算预算。当前不支持AnyFlow32的4步画质优势，FM本身也未过动作gate。
