# EXP-024/v1 Worker 报告：V3 普通 FM 训练覆盖与容量准备

2026-10-15 HKT。**CPU 阶段完成，待 Judge 验收；0 GPU、0 真实 H3 forward/backward/optimizer update、0 视频。** 小张量 CPU 测试执行了 toy optimizer 更新。本任务的判断对象是下一轮训练设计，不是模型能力。执行依据：[任务书](taskbook_v1.md)和 [CPU-only 授权](AUTHORIZATION.json)。EXP-023 的正式结论和原始证据未修改。

## 对 EXP-023 的容量与变化审计

只读取 checkpoint0/128，未加载 33B 基座。[逐层审计](PARAMETER_AUDIT.json)验证 checkpoint SHA、8 个 adapter 与真实模块形状：50 层 DiT 中，仅层 42–49 各有一个 QKV LoRA，A 为 `[8,5376]`、B 为 `[21504,8]`，推理增量 `ΔW = BA/8`。checkpoint0 的 B 为零；128 更新后 8 层 B 均非零。通过低秩 Gram 公式计算 `||ΔW||F`，避免物化巨大的 QKV 矩阵；8 层数值在 **0.4021–0.4702**，对单位各向同性输入的等效输出 RMS 代理为 **0.00548–0.00641**。A/B 参数确实改变，但这些仅是权重空间量，不含真实 hidden-state 分布与输出画质，不能将 EXP-023 的视频收益不足直接归因“容量不足”。

| 候选 | 可训练参数 | FP32 参数+梯度+Adam m/v 静态合计 | 相对 EXP-023 |
|---|---:|---:|---:|
| tail8 rank8（EXP-023） | 1,720,320 | 27,525,120 B（26.25 MiB） | 1× |
| full50 rank8 | 10,752,000 | 172,032,000 B（164.06 MiB） | 6.25× |
| full50 rank32 | 43,008,000 | 688,128,000 B（656.25 MiB） | 25× |

静态计算使用真实 QKV `5376→21504`，每层参数 `rank×(5376+21504)`，FP32 训练常驻按参数4B、梯度4B、Adam一/二阶各4B估算。它**不**包含 33B 基座、released Action LoRA、激活、checkpoint 重算、CPU raw KV、分配器碎片和临时算子。EXP-023 tail8 的真实 GPU peak allocated 24.540 GiB、C2 CPU raw KV 6,799,104,000 B；这些数字也不能线性外推 full50 backward。全层适配将使早层成为可训练路径，预期 checkpoint 反向重算覆盖从旧 tail8 的层49→42扩大到全部50层；真实 h3_fp32/offload 的时间与峰值仍须测。

## C1/C2 实现与 CPU 证据

[独立训练入口](coverage.py)明确选择 `C1` 或 `C2`。C1 对12个 latent 计算普通 FM loss，历史 cache 为空；C2 只读取同 clip 的 GT C1，经 σ=0/no-grad clean commit 后训练5个 latent。两者都通过 `visible_inputs` 在调用 refiner/DiT 前移除未来动作与视频；`interval_train` 检查 native global RoPE 与布局；`verified_attention_scope` 覆盖完整 forward 和 checkpoint backward；训练时 raw KV 不被反向修改。loss 对每个块的 latent 元素取 mean，未来拟合顺序由 `balanced_block_order` 保证每 clip 每 epoch 恰有一次 C1 和一次 C2，避免12:5长度隐含加权。C2 以当前 adapter 重新 commit GT C1，不能跨 optimizer 更新复用旧 KV。

- [CPU 小张量测试](CPU_TEST_RESULTS.json)调用实际 `visible_inputs`、`interval_train`、`ChunkAttention` 和 `verified_attention_scope`，以 mock DiT 代替 33B：C1 无历史，C2 clean commit 有历史；两块前向与 checkpoint 反向均使用 current-prefix，自己的 action 可见、直接过去动作不可见，future prompt/video 扰动不影响 C1。两块 adapter 梯度有限且非零、冻结 toy base 无梯度；序列化 adapter/optimizer/RNG 后，固定下一步 loss 与参数更新逐值一致。
- [50层 tiny checkpoint 测试](FULL_LAYER_CPU_RESULTS.json)使用50个微型 adapter，各层真实 `ChunkAttention` 在 C1/C2 forward 和 backward 重算都处于正确 context；C2 50层 GT clean commit 已记录。它验证作用域生命周期，**不模拟真实33B激活/卸载峰值**。
- [34条真实条件 CPU 审计](REAL_INPUT_AUDIT.json)逐条加载 EXP-023 编码的 condition，C1 可见12个 action spans/C2可见17个，sequence行数等于 prefix+对应 video rows；Single I0 anchor为390行。未用旧 H.264 latent 或新数据。
- [资源复测](CPU_RESOURCE_AUDIT.json)四个脚本顺序运行，设定4线程、`CUDA_VISIBLE_DEVICES=-1`；单进程最高 RSS 964,468 KiB，远低于额外8GiB限制。[守卫CPU测试](PROFILE_GUARD_CPU.json)确认现有 CPU 授权被 [profile 入口](profile_gpu.py) 在接触 CUDA 前拒绝；本轮没有创建 GPU profile 结果。

## 唯一建议与下一门槛

首选 **full50 rank8 + C1/C2 同权普通 FM**，将其明确写作相对 EXP-023 的联合候选，不声称是层数或训练覆盖的单变量消融。[具体 GPU profile 提案](GPU_PROFILE_PROPOSAL.md)与 [守卫入口](profile_gpu.py)规定新 Judge 授权后单张空闲 L40、最多 C1/C2 各1次更新、预期3顶层 DiT/2 backward/2 update、每阶段同步计时与显存/RSS、50层路由/冻结梯度检查及44GiB/128GiB/授权时长硬停。GPU profile **尚未执行**；当前 CPU 授权 `gpu_authorized=false`。若真实两步 profile 通过，才由 Judge 决定是否另外授权 32 clip×C1/C2 各一次的64-update 小数据拟合，且必须依据测得的 C1/C2 成本核算预算；不自动延长到128或256更新，也不转 AnyFlow/DMD。

本轮没有新视频和质量测量。即使下一次两步 profile 通过，也只证明训练可执行；C1/C2 联合训练的动作、视觉和自生成历史能力仍要用匹配的固定视频与同σ probe 判断。新代码与来源的 SHA 见 [清单](SOURCE_MANIFEST.json)。Worker 未提交或推送 Git。
