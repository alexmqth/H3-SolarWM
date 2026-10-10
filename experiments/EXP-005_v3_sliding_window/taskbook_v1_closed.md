# EXP-005 / v1：V3 Sliding Window 准备与分阶段验证

2026-10-10，Judge依据用户最新要求发布。本轮只授权阶段一CPU工作，GPU预算当前为0；阶段二任务书提交后等待独立Judge批准，不能以本任务编号或CPU通过作为GPU授权。

## Task / Track / Parent / Research Question

- EXP-005 / plan_version 1；Worker completed（CPU）；Judge accept（仅阶段一）；阶段二GPU pending / NOT AUTHORIZED。
- Mainline systems / positional protocol；Parent为EXP-002/003冻结V3 Original Feasibility Baseline，30步、Global RoPE、strict causal/persistent raw KV、124帧。
- 问题：能否在首淘汰前保持冻结V3行为，真实淘汰后只保留最近5祖先，并用独立Local位置候选区分窗口与位置影响？
- V3 Baseline保留已验收结果。V3-SW-G优先；V3-SW-L独立候选，不预设无损或更优。此前短暂的V2c分类被本次用户最新命名取代。

## 固定条件 / 允许变化

保持Original H3 + released Action LoRA、Single I0、native timestep（video sigma×1000，audio1000，fixed_prefix_timesteps=False）、own-action/action feedback/current non-action prefix feedback、sigma0 clean commit、partition12后每块5、seed/噪声与30步native FM shift2.22不变。

SW-G仅泛化显式分块区间与最近5祖先缓存检查。SW-L额外改变明确的video位置映射；不能改prefix、timestep或动作mask来救质量。当前H3是否有camera PRoPE必须以冻结源码核实，不照搬SolarWM。

## 阶段一：CPU授权

1. 阅读冻结EXP-002/003/004、实际runtime h3_cached/current_prefix/interval与SolarWM sgf_attention/sgf_rollout，记录源码hash。
2. 在EXP-005独立目录实现显式ChunkPlan与interval入口，支持index>5；验证每层indices恰为range(max(0,k-5),k)，拒绝缺层/空cache伪通过，不改已验收文件。
3. 复用H3ChunkCache(max_history=5)，测试真实commit淘汰、容量上界（首12与后5长度不同）、祖先raw KV不变、sigma0提交与历史RGB append-only。
4. SW-G复用冻结current-prefix数学/条件；CPU测试旧支持区间2–5的调用协议与真实attention数值等价，明确CPU fp32/backend与未做完整33B输出回归。
5. SW-L明确native时间网格、raw K/V读取时重映射、canonical位置存储、prefix不变、camera处理；考虑非均匀5周期时间网格，不用latent index简单减法或猜测常量时间偏移。CPU验证实际H3 RoPE及prefix跨模态影响；不能声称冻结raw KV重定位等价整段重算。
6. 验证未来action/video物理隔离、global/local索引、错误cache拒绝及长输入构造前37latent/原prefix位置不变。若尚无安全的>37latent真实输入，显式标为阶段二审批前的必备项，不静默新建改变首窗的全长fixture。
7. 只加载CPU小型fixture/tensor；不加载33B权重、LoRA GPU、不发模型诊断、不启动任何GPU作业。CPU最多4线程，避免大cache克隆。

## 预算 / Stop / Out of Scope

当前：0 GPU forward、0GPU小时、0VAE、0训练；CPU小张量/已保存条件检查。协议有歧义先记录并定位，不以改time/prefix/mask绕过。保留冻结baseline所有代码/指标/报告/视频，不覆盖现有正式结果。

不启动LoRA训练、RGB Dual Anchor、AnyFlow/DMD、GPU推理或新场景/seed扫描。阶段二预算只是提案，最终以独立明确授权为准。

## 交付与阶段二准备

- README / PROTOCOL / CPU代码与测试日志、source manifest与指标、风险说明。
- 分阶段GPU草案：先首淘汰前回归，再第7/8块Global与Local固定历史/动作/噪声配对；第9块作为事先写明但默认不启用的选项。
- 实际淘汰indices、KV字节、旧RGB冻结、A/D与切换、全帧视觉、采样/commit/VAE/NFE/峰值/wall等验收与明确停止条件；CPU通过不代表生成能力通过。
- 独立V3-FM8与V3-AF未来方案，匹配NFE/历史和target-time-conditioned student；本任务不执行训练。
- Worker负责实验目录的CPU实现、测试、协议工程说明和根report；Judge负责命名/正式导航、最终预算提案与审核/Git交付。阶段一完成后停止等待。

## Judge验收与当前交接状态

2026-10-10：接受阶段一CPU实现/协议交付。最终14项测试通过，Judge独立复跑14 passed in 3.74s；具体范围见[审核](submission/experiments/EXP-005_v3_sliding_window/judge/FINAL_REVIEW.md)。旧index2–5在CPU fp32真实attention路径逐元素一致，模型实际传参与冻结入口一致；不等于33B完整输出或生成能力验收。

**当前0 GPU / 0训练授权继续有效。** SW-G/SW-L能力NOT_TESTED，V3 Baseline正式证据冻结。Worker本轮收口，不自行开启下一阶段。

阶段二审批前先准备经认证的>37latent原生fixture与独立预算runner：保持原37坐标/条件/prompt/noise，显式冻结新增action与位置；当前toy长输入不可用于真实模型。此项尚未完成，GPU入口保持拒绝。

[GPU提案](submission/experiments/EXP-005_v3_sliding_window/GPU_PLAN.md)：G0→G1→L1分别审批，合计281forward/9VAE/1.70GPU小时上限，仅1卡、项目≤3；C9默认关闭。GPU回归需记录精度/backend与误差；动作、结构、边界、历史不变及系统成本分别验收，不能凭finite forward判PASS。

[独立FM8/AnyFlow计划](submission/experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md)已提交，仅设计，不执行训练。
