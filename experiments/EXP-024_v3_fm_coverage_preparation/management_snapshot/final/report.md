# EXP-024/v2 Worker 报告：V3 普通 FM 训练覆盖与全层 GPU profile

2026-10-15 HKT。**v1 CPU 准备与 v2 单卡两更新 profile 均完成，待 Judge 验收；0 视频。** CPU 阶段为0 GPU、0真实 H3 更新；随后按 [v2任务书](taskbook_v2_gpu_profile.md)和 [GPU授权](GPU_AUTHORIZATION.json)执行一次真实33B profile。没有扩展成32样本训练，也未修改 EXP-023 正式证据。

## 对 EXP-023 的容量与变化审计

只读取 checkpoint0/128，未加载 33B 基座。[逐层审计](PARAMETER_AUDIT.json)验证 checkpoint SHA、8 个 adapter 与真实模块形状：50 层 DiT 中，仅层 42–49 各有一个 QKV LoRA，A 为 `[8,5376]`、B 为 `[21504,8]`，推理增量 `ΔW = BA/8`。checkpoint0 的 B 为零；128 更新后 8 层 B 均非零。通过低秩 Gram 公式计算 `||ΔW||F`，避免物化巨大的 QKV 矩阵；8 层数值在 **0.4021–0.4702**，对单位各向同性输入的等效输出 RMS 代理为 **0.00548–0.00641**。A/B 参数确实改变，但这些仅是权重空间量，不含真实 hidden-state 分布与输出画质，不能将 EXP-023 的视频收益不足直接归因“容量不足”。

| 候选 | 可训练参数 | FP32 参数+梯度+Adam m/v 静态合计 | 相对 EXP-023 |
|---|---:|---:|---:|
| tail8 rank8（EXP-023） | 1,720,320 | 27,525,120 B（26.25 MiB） | 1× |
| full50 rank8 | 10,752,000 | 172,032,000 B（164.06 MiB） | 6.25× |
| full50 rank32 | 43,008,000 | 688,128,000 B（656.25 MiB） | 25× |

静态计算使用真实 QKV `5376→21504`，每层参数 `rank×(5376+21504)`，FP32 训练常驻按参数4B、梯度4B、Adam一/二阶各4B估算。它**不**包含 33B 基座、released Action LoRA、激活、checkpoint 重算、CPU raw KV、分配器碎片和临时算子。EXP-023 tail8 的真实 GPU peak allocated 24.540 GiB、C2 CPU raw KV 6,799,104,000 B；这些数字不能线性外推 full50 backward。v2已对真实 h3_fp32/offload 两步作有限实测，结果见下。

## C1/C2 实现与 CPU 证据

[独立训练入口](coverage.py)明确选择 `C1` 或 `C2`。C1 对12个 latent 计算普通 FM loss，历史 cache 为空；C2 只读取同 clip 的 GT C1，经 σ=0/no-grad clean commit 后训练5个 latent。两者都通过 `visible_inputs` 在调用 refiner/DiT 前移除未来动作与视频；`interval_train` 检查 native global RoPE 与布局；`verified_attention_scope` 覆盖完整 forward 和 checkpoint backward；训练时 raw KV 不被反向修改。loss 对每个块的 latent 元素取 mean，未来拟合顺序由 `balanced_block_order` 保证每 clip 每 epoch 恰有一次 C1 和一次 C2，避免12:5长度隐含加权。C2 以当前 adapter 重新 commit GT C1，不能跨 optimizer 更新复用旧 KV。

- [CPU 小张量测试](CPU_TEST_RESULTS.json)调用实际 `visible_inputs`、`interval_train`、`ChunkAttention` 和 `verified_attention_scope`，以 mock DiT 代替 33B：C1 无历史，C2 clean commit 有历史；两块前向与 checkpoint 反向均使用 current-prefix，自己的 action 可见、直接过去动作不可见，future prompt/video 扰动不影响 C1。两块 adapter 梯度有限且非零、冻结 toy base 无梯度；序列化 adapter/optimizer/RNG 后，固定下一步 loss 与参数更新逐值一致。
- [50层 tiny checkpoint 测试](FULL_LAYER_CPU_RESULTS.json)使用50个微型 adapter，各层真实 `ChunkAttention` 在 C1/C2 forward 和 backward 重算都处于正确 context；C2 50层 GT clean commit 已记录。它验证作用域生命周期，**不模拟真实33B激活/卸载峰值**。
- [34条真实条件 CPU 审计](REAL_INPUT_AUDIT.json)逐条加载 EXP-023 编码的 condition，C1 可见12个 action spans/C2可见17个，sequence行数等于 prefix+对应 video rows；Single I0 anchor为390行。未用旧 H.264 latent 或新数据。
- [资源复测](CPU_RESOURCE_AUDIT.json)四个脚本顺序运行，设定4线程、`CUDA_VISIBLE_DEVICES=-1`；单进程最高 RSS 964,468 KiB，远低于额外8GiB限制。[守卫CPU测试](PROFILE_GUARD_CPU.json)确认旧 CPU 授权被 [profile 入口](profile_gpu.py) 在接触 CUDA 前拒绝；v2新授权另行签发。
- v2 加固：`coverage.cache_integrity` 同时比较 EXP-023 `cache_signature` 的对象/形状/首8项采样，以及**全部已提交CPU K/V/RoPE payload**的 SHA256；不把抽样签名单独称为全内容等价。[故障注入](GATE_FAULT_CPU.json)在同字节数且旧签名不变时修改第20个KV值，完整摘要于optimizer前拒绝；注入缺层路由和冻结权重梯度也使optimizer调用数保持0，C1失败不进入C2。50层微型正常路径的每块optimizer前门控亦通过。这些检查不覆盖模型临时GPU激活。

## v2 真实33B单卡两更新 profile

启动前确认 GPU0 空闲（L40，45,458 MiB free），只使用该卡；冻结源码/输入 SHA、授权和命令/PID/退出码在 [启动记录](GPU_PROFILE_LAUNCH.json)及 [结果](GPU_PROFILE_RESULT.json)，完整日志在 [GPU_PROFILE.log](GPU_PROFILE.log)。外层 `timeout 1770s + kill-after 10s`，实际进程 **76.367秒、exit 0**，无重试。模型为 Original H3 + released Action LoRA 冻结，新零增量 full50 QKV rank8；同一固定训练 clip，C1/C2 各1次普通 FM 更新，σ=0.5、独立CPU noise RNG130024、seed13，按 v2 原协议执行。

| 阶段 | 同步耗时 | 阶段末峰值 allocated | 阶段末峰值 reserved |
|---|---:|---:|---:|
| C1 forward | 17.855 s | 23.043 GiB | 27.342 GiB |
| C1 backward | 14.427 s | 24.163 GiB | 27.383 GiB |
| C1 optimizer | 0.056 s | 24.163 GiB | 27.504 GiB |
| C2 GT C1 clean commit | 11.186 s | 24.163 GiB | 27.504 GiB |
| C2 forward | 7.740 s | 24.163 GiB | 27.504 GiB |
| C2 backward | 10.529 s | 24.163 GiB | 27.504 GiB |
| C2 optimizer | 0.005 s | **24.163 GiB** | **27.508 GiB** |

进程所记录最高 RSS **55.737 GiB**；C2 CPU raw KV **6,799,104,000 B**。累计计数为 **3/3顶层DiT尝试且完成、2/2 backward、2/2 optimizer**，0 VAE/text/视频。C1 loss 0.120413、grad L2 0.0004537；C2 loss 0.121229、grad L2 0.0005445。两块状态/σ/目标不同，两个 loss **不能**构成学习趋势或质量结论。

每块在optimizer前检查了50层当前块forward及checkpoint backward的 `current_prefix` 路由；C2额外检查GT C1 commit 50层、历史KV结构与全内容摘要不变。50组新增adapter梯度均存在且有限、总norm正，冻结 base/released Action LoRA无梯度。最终 route trace 250条，完整层列表、逐层grad及尝试/完成计数保存在结果JSON。precision 记录为 `h3_fp32`（输入/时间条件FP32、Transformer保持所加载精度），并保存固定clip的 condition/full17/I0 SHA。结果只证明本候选在这一条clip上可完成两步反向与更新，**不证明画质、动作、长期自生成历史或训练吞吐的稳定分布**。

## 唯一建议与下一门槛

首选 **full50 rank8 + C1/C2 同权普通 FM**，将其明确写作相对 EXP-023 的联合候选，不声称是层数或训练覆盖的单变量消融。两步profile在资源和梯度协议上通过，尚未训练可评画质的模型。[具体执行协议](GPU_PROFILE_PROPOSAL.md)与 [守卫入口](profile_gpu.py)可复核。以本次 C1+C2 阶段合计约61.80秒粗估，32对/64更新的纯更新成本约1978秒；加此前约1500秒固定前后评测及20%缓冲约4174秒。单clip首轮耗时不能代表32clip稳定吞吐，后续预算应由Judge按此作为**规划估算**另行批准；不能自动延长训练、转AnyFlow/DMD或升级V3能力状态。

本轮没有新视频和质量测量。C1/C2 联合训练的动作、视觉和自生成历史能力仍要用匹配的固定视频与同σ probe 判断。新代码与来源的 SHA 见 [清单](SOURCE_MANIFEST.json)。Worker 未提交或推送 Git。
