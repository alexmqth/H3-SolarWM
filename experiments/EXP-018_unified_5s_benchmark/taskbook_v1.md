# EXP-018/v1 — V0–V4统一约5秒视频与推理成本基准

2026-10-13 HKT，Judge。用户明确要求V0、V1、V2、V3、V4各生成同长度视频，并完整对比生成耗时、采样step/NFE和GPU峰值显存。本任务取代EXP-017已完成的任务书，授权下述有限推理；无训练。

## 统一输入和长度

停车场同一I0源图、scene prompt、持续A、seed13、832×480、24fps、flow shift2.22。保留各版本定义的anchor/attention/adapter/precision和chunk partition，不把旧版本改成V3。原生H3长度为5+17k，各行真实生成124帧（5.1667秒），同时导出前120帧严格5.000秒展示片。所有成本明确为完整124帧生成，不冒充120帧测量。保留124原片；不得复用历史生成首窗、endpoint或KV。

## 七个冻结配置，各单次从噪声完整生成

1. V0 Original H3，released Action LoRA，整片30NFE。
2. V1 Native causal，历史代表dynamic_last_frame_dual，无新adapter，5latent/chunk，8NFE/chunk。
3. V2a RGB Anchor，冻结visual/action适配器，dynamic_last_frame_rgb_dual，5latent/chunk，8NFE/chunk。
4. V2b Same-sigma local bidirectional，冻结native条件，12+5分块，30NFE/chunk，无persistent KV。
5. V3 Baseline strict causal raw KV Global，冻结current-prefix/own-action，12+5分块，30NFE/chunk。
6. V3 FM-8NFE候选，同V3协议，首窗也新8NFE生成；124帧不声称验证滑窗淘汰。
7. V4-DMD，EXP-008最终cycle8 student QKV/target-time配套权重，8NFE/chunk，V3 causal协议。本次授权从首窗开始的独立评估扩展。旧配置曾失败，不挑checkpoint、不训练修复；数值有限即使质量差也完成真实124帧、如实FAIL，不用借来的FM首窗/循环/冻结帧补时长。若原协议不支持从首窗，报告具体UNSUPPORTED，不擅改网络。

## 预检与公平口径

CPU预检：权重/来源/SHA、版本协议、路径迁移、fresh-noise/no-reused-endpoint、计数器与显存统计、GPU空闲门。新独立wrapper可复用冻结模块，不覆盖旧源码和产物。可统一共享输入编码fixture，但共同准备需单独真实测一次；不能将fixture加载冒充编码成本。各行记录初始noise SHA，shape不同时明确不能逐值比较。

优先同一空闲L40串行，统一offload显存预算。记录各行backend/precision/UUID/编译与初始空闲状态；各版本必须的精度差异明确披露。本任务默认1卡，09:00后最多3卡。00:32仅GPU5空闲，其它GPU有任务，启动前重查、不抢占。

每行独立进程，完整首窗至124帧：model/adapter load、condition prepare或fixture load、sampling、KV prefill/clean commit、decode、encode/save、ready-to-video、process-to-final-file总wall分别记录。ready-to-video从模型和条件ready起到124RGB完成；cold total从进程启动到最终124帧文件落盘，含加载/准备。按CUDA同步阶段边界。排除人工review等待，不跨行复用生成cache。共享编码耗时单列，需计入可比总成本时明确折算。

从CUDA初始化起记录全流程max allocated/reserved；reset阶段峰值时保留含加载/编码/采样/commit/decode的全局最大。NVML采样若实现则另列。CPU KV峰单列。分别记录sampling_steps_per_chunk/full_video_steps、sampling_NFE_total、commit/prefill_forward_total、encode/decode次数；8NFE/chunk不等于全片8NFE。

本轮为冷启动单次方案基准n=1，未消除编译/系统波动，不声称统计稳定排名。旧复现文档的warmup+3repeat不适用于本轮ROI有限对比。

## 预算、停止、授权

7行各1次正式推理；总4 GPU小时、单行45分钟、显式sampling/commit/prefill总forward上限850，0训练。实现/导入错误在模型forward前可修复；GPU后错误必须保留账本，最多1次明确原因修复重试、仍计总预算；不因画质差筛seed/权重/采样次数。

每行启动检查无人占用该卡；遇竞争/显存不足暂停后续启动，只管理EXP-018自身进程。非finite/OOM/权重配对错误/预算超限停止该行，报告真实失败；质量FAIL不提前截断此次124帧评估。不得额外动作/scene/seed/AF扫描。4CPU affinity；不逐块保存大型KV，新增磁盘≤10GiB，空余不足40GiB停止。

本任务书即Judge对7行GPU授权。Worker完成CPU预检并保存具体输入/source manifest和预算配置后可在空闲GPU启动，无需重复批准。

## 交付

submission/experiments/EXP-018_unified_5s_benchmark/保存源清单、日志、统一JSON/CSV、命令/环境/权重hash、124原片和120展示片、统计表及失败/协议差异。运行输出H3-World/outputs/EXP-018_unified_5s_benchmark/。每行完成原子状态；更新report.md。120帧/24fps/5秒展示片完整解码、PTS/来源检查；执行成功不等于画质通过。

Judge核查七行权重/新首窗、计时/峰值范围和视频，制作完整并排总览、发布GitHub。旧正式V3及历史负结果保持。
