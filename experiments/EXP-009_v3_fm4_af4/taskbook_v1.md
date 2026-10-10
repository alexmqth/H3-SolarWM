# EXP-009 / v1 — 普通FM4与AnyFlow AF4有限续写对照

2026-10-11，Judge。**当前仅CPU准备；GPU须Judge核对实现及冻结来源后发独立授权marker。** 用户夜间持续研究授权有效至09:00 HKT，无需再次征求批准。EXP-008当前DMD配置生成FAIL并归档，不恢复AD、不追加训练或换checkpoint。

## 问题、假设与基线

Research Track C；Parent为冻结V3因果协议与EXP-007 AF2 step32。唯一问题：已在8NFE可用但没有整体优势的AnyFlow student，在4NFE是否比普通FM4更能保留结构与动作？有限推理可以直接改变是否值得继续AnyFlow训练的决策。假设未预设成立。

两候选：Original H3 + released Action LoRA普通FM4；同backbone加AF2 step32新QKV与target-time的AF4。复用已有FM8/AF8为8NFE参考，不重跑。不加载EXP-008 DMD权重。模型/训练方法不同，按方案比较，不能宣称训练目标单因素归因。

## 固定条件和输入

共同EXP-006 FM8首12 clean latents与前39 RGB，seed13冻结初噪声的C2/C3 slices；AA与AD动作来自冻结parking_A/D。每模型独立prefill自己的C1 raw KV。C2在同clean历史比较；C3各用自己的C2，不交叉借用KV。明确这是共同FM8首窗后的4NFE续写，不是全程FM4/AF4首窗生成。

保持Single I0、native timestep（外部1000sigma/内部1-sigma）、audio1000、own-action/action feedback、current non-action prefix feedback、fixed_prefix_timesteps=False、Global RoPE、strict causal、persistent CPU detached raw KV、max_history=5、12+5+5 partition、clean sigma0/r0 commit、h3_fp32和既有attention backend。

AF2 checkpoint：H3-World/outputs/EXP-007_v3_anyflow_af2/step_32/。QKV SHA 07c8e5e68d1c947e60217e326ca8dd4c00222a555542b286dfc03514e27e916e；target-time SHA ceb7d62324834a5b2be9fd6a47bfe4b5a9d810e49d1bb33889af0fb788940f70。last8 blocks42–49/rank8/scale.125/target gate.25。仅已有student，无新训练。

## CPU实现和验证

1. 复用已审计EXP-006/007/008推理骨架，4-step grid必须从原生MiniMax-H3 scheduler、shift2.22生成，不能猜测或保留8-step硬编码。
2. FM4每步普通velocity更新；AF4每步target r为下一个native sigma。验证4次调用、终点0、正确time/r、clean commit和prefix约束。
3. 固定source/config/input/checkpoint SHA；验证各模型自身KV、历史indices、共用噪声/动作以及已发布RGB不变性。CPU预检用CUDA_VISIBLE_DEVICES为空，不加载GPU。
4. 给出实际运行命令、预算扣减/失败保留、GPU超时与锁、日志/metrics/代表视频路径；无自动重试。

## 有限GPU阶段预算（待Judge独立放行）

每模型prefill C1一次、AA/AD C2/C3各4sampling、C2各clean commit一次，合计19forward、4VAE。总上限**38forward、8VAE、0backward、0update、累计0.60GPU小时**，单卡allocated≤44GiB，最晚09:00 HKT。预算包括加载/保存/等待与失败，不能按重启清零。两个模型可在两张实际空闲卡上独立执行，实际GPU选择写入marker；GPU3/4他人进程禁止抢占。09:00后项目总占用最多3卡。

执行顺序每模型prefill→AA/AD C2→Judge看C2→AA/AD C3；AA C2出现全段噪声/主体场景持续崩溃时，该模型停止后续启动；已经在途的另一候选先完成当前获批块再评估。普通局部缺陷不额外卡住。完整至73帧只有两个动作分支，不加scene/seed/NFE/checkpoint sweep。

## 验收与停止条件

记录全部新增帧、A/D方向与切换、人物与停车场结构、ghosting、chunk boundary；光流只辅助视觉判断。逐块给出sampling/commit/decode时间、GPU allocated峰值、CPU KV bytes、实际forward/VAE数及可复现命令。首39 RGB逐值不变、缓存来自自身权重、无未来泄漏。

在共同首窗上提供FM4/AF4完整AA/AD对比，并用已有8NFE片辅助看降步损失。能力结论与工程PASS分开；普通画质缺陷允许PARTIAL。无需成熟画质门槛，不因轻微proxy差异追加实验。

OOM/nonfinite、错误checkpoint/时间/cache、预算或截止触顶立即停止，无自动重试。持续全画面噪声或主体/场景消失则该配置FAIL并收口，不继续C3凑满预算。AF4若没有有意义的联合收益，停止当前短训student的NFE探索，不自动追加2NFE、训练或DMD挽救。

## 交付

EXP-009目录内保留README/config/source manifest、CPU结果、GPU授权、脚本、真实原始日志/账本、指标/视频与Worker报告；Judge审核后更新progress/next_plan/report并发布GitHub。大型checkpoint、KV与latent保留外部路径和SHA。正式V3 Baseline与既有结果冻结。当前禁止新GPU启动，CPU准备提交后由Judge发marker。
