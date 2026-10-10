# EXP-005 / v2 Worker 执行报告：V3 Sliding Window

`task_id=EXP-005` · `plan_version=2` · `worker_status=completed` · `judge_final_acceptance=accepted_with_quality_partial`

更新：2026-10-10 HKT。阶段一CPU已由Judge验收；本报告记录阶段二的真实执行。旧v1报告保存在[阶段二快照](stage2/previous_report.md)，冻结V3 Baseline/EXP-003结果不覆盖。**G0、G1、L1均已执行并获Judge按各自范围验收；Local画质PARTIAL，无相对Global收益，未升级候选。**

## 核心问题和实验边界

V3 Baseline 的 strict causal + persistent raw video KV 在124帧已具有限单场景/seed可行性。本轮验证首次真实窗口淘汰及Local RoPE候选是否能把续写延到第7/8块，同时保持动作方向与人物/场景基本结构。全程仍用 Original H3 + released action LoRA、Single I0、own-action/current-prefix feedback、30步native FM shift2.22；没有训练、AnyFlow或DMD。长输入是经CPU认证的47-latent显式扩展，不等于原H3默认整段重打包。

## G0：首淘汰前完整模型回归，Judge已验收

两组同状态old interval与SW-G velocity完全一致（sigma1和0.5的maxabs/relativeRMS均0）；30步C6重放endpoint与冻结EXP-003逐值相等；124RGB全段逐像素相等，旧107帧未变。实际4诊断+30采样=34forward、1VAE、309.111s、峰值allocated25.598GiB；0训练。视频124帧、24fps完整可解码。详见[G0结果](stage2/G0_RESULT.md)和[Judge审阅](stage2/g0_judge_review.json)。这只能证明旧区间回归，不能推断淘汰后生成能力。

## G1：SW-G真实淘汰，已执行完成并获有限可行性验收

EXP-003 C6 clean endpoint在全部50层恰好提交一次后，C7的实际祖先为`[1,2,3,4,5]`，各分支C7提交后C8为`[2,3,4,5,6]`。A继续与D切换在**同一C6生成历史、同一新增噪声**下开始；C8各接自己的C7输出。第7块同状态A/D的水平光流分别为+0.819/−1.575 px/帧，差值约2.394；第8块各自闭环为+0.768/−1.185。人物和停车场基本可辨，但A-C8有短暂手臂/颜色拖影，D-C7切换边界跳变明显，D-C8白色拖影仍存在。因此动作方向为正信号，画质/连续性PARTIAL，不能只据flow判质量通过。

已保存两条158帧原片、四段17新增帧原片、全新增帧contact sheet及[带标签并排视频](artifacts/stage2/G1/G1_A_vs_D_158.mp4)；141/158帧视频全部24fps、832×480逐帧可解码，所有已显示RGB前缀逐值不变。实际120采样+3提交=123forward、4VAE、1068.666s、peak allocated26.121GiB、reserved26.543GiB、process RSS峰值约91,512MiB。历史video raw KV淘汰后为14,164,800,000 bytes，不能称整个系统内存有界。详细表格、视频与证据见[G1结果](stage2/G1_RESULT.md)、[机器指标](artifacts/stage2/G1/result.json)和[Judge审阅](stage2/g1_judge_review.json)。G1进程已退出；无自动重试或C9。

## L1：SW-L Local位置对照，已验收执行证据、画质PARTIAL

独立的SW-L runner/配置先通过CPU Local全局/局部固定状态、clean commit淘汰及预算拒绝超限。Judge审核G1完整视频后以[l1_authorization.json](stage2/l1_authorization.json)绑定runner/config/manifest批准L1。L1共享G1 C6缓存，先在同一G1-A C7历史/C8 noisy state比较Global和Local两次只读velocity，再各自生成A/D的C7/C8。固定状态下位置改变的velocity `max_abs=2.0033`、`relative_RMS=0.1825`；历史未改。这只证明位置有数值影响。

C7相同生成历史和噪声下，Global的A/D水平光流为`+0.819/−1.575`，Local为`+1.120/−0.598 px/帧`；separation由`2.394`降到`1.718`。C8每路接自己的C7，非同状态对照；Local A/D仍为`+0.343/−1.269`，但不能由此声称动作或质量更好。Local四段帧内灰度MAD均约`9–10`，Global约`4.6–4.9`；逐帧看见Local停车场透视重组、亮度跳变、彩色边缘及人物附近残影。人物和场景仍可辨，但**没有显示Local相对Global的视觉收益**。部分boundary MAD下降也不足以推翻完整画面判断。

两条Local原片与[Global-vs-Local A](artifacts/stage2/L1/G1_vs_L1_A_158.mp4)、[Global-vs-Local D](artifacts/stage2/L1/G1_vs_L1_D_158.mp4)并排视频均逐帧解码通过：158帧、24fps，原片832×480，并排1664×560；旧RGB前缀逐值不变。L1实际120采样+2提交+2诊断=`124forward`、`4VAE`、`1007.969s`、GPU0 peak allocated`26.621GiB`、reserved`26.994GiB`，进程RSS峰值约`104,275MiB`。GPU0进程已退出，无重试、C9或训练。详细见[L1结果](stage2/L1_RESULT.md)、[Judge验收](stage2/l1_judge_review.json)、[机器指标](artifacts/stage2/L1/result.json)及[预算](artifacts/stage2/L1/budget.json)。G1/L1阶段耗时包含不同诊断、commit和I/O，不能当成公平E2E加速比较。

## Worker结论与交接

EXP-005/v2已完成并通过Judge执行证据验收，合计`281forward/9VAE/2385.746s≈0.663 GPUh`、0新增训练。G0证明旧区间回归；G1提供真实W5淘汰后的158帧动作/结构有限可行性，画质PARTIAL；L1验证Local位置明显改变velocity，但在现有单场景seed下没有动作/画质联合收益。保持正式V3 Baseline冻结，SW-G是本轮更强的有限续写证据，SW-L作为机制负结果归档、当前无训练Local方向停止。Judge不批准C9或继续调参；后续新任务需独立立项。
