# EXP-024/v1：普通 FM 基线的训练覆盖与容量准备

2026-10-15，Judge 授权立即执行 CPU 阶段。Research Track：Branch B；Parent：冻结 V3 与 EXP-023 修复后的训练入口。

## 研究问题与决策

EXP-023 完成128updates，固定 FM probe 略降但视频收益未证实。下一次有效基线应该覆盖什么训练块、采用多大 adapter，如何在现有 L40 上有限验证？本轮交付可执行实现与一个建议方案，不做参数搜索。假设：只训 GT-history C2、tail8 rank8 的最小 profile 容量不足以代表完整 causal adaptation；这是待验证假设，不能预先归因容量不足。

## 冻结条件与变量

Original H3 + released Action LoRA 冻结；Single I0/native time/Global RoPE/own-action/action feedback/current-prefix feedback/clean raw KV/12+5 partition/SDPA/h3_fp32 不变。复用 EXP-023 的32train+2dev编码与6个视频评测角色、相同噪声规范。普通 FM，非 AnyFlow。允许扩展训练块覆盖（C1和C2）和候选 adapter 层/rank；两者若同时改变须标为联合基线方案，不能声称单因素消融。EXP-023 作为最小容量参考。

## 立即执行

1. CPU 读取 checkpoint0/128，量化各层 adapter 参数变化、低秩增量作用量；不要加载33B完整模型，避免将参数差异当作输出改善。
2. 在本实验目录新增训练准备代码，支持显式 C1 或 C2 loss。C1无历史；C2使用GT C1无梯度clean commit；context覆盖完整forward/backward。定义块采样与loss归一化，避免训练时未来视频/动作泄漏。复用生产路径做有意义的小张量CPU梯度/隔离/保存恢复检查；不得修改冻结EXP-023源码。
3. 从真实模块形状计算 tail8 rank8、全层QKV rank8/rank32 的参数量和optimizer静态成本，分清静态估算与尚未实测的激活/反向峰值。检查全层反向对offload/checkpoint生命周期的影响。选择一个首选方案，不同时铺开多个训练分支。
4. 给出下一GPU profile的具体入口、最多2次更新（C1/C2各一次）、逐阶段forward/backward/显存/耗时记录、硬停和预计后续32样本overfit预算。GPU profile须另经Judge放行；不能把CPU测试写成33B反向已通过。
5. 写 WORKER_REPORT.md、CPU_TEST_RESULTS.json、PARAMETER_AUDIT.json、GPU_PROFILE_PROPOSAL.md；同步根report.md，立即消息通知Judge。

## 预算与停止条件

CPU-only：最多4 CPU线程、30分钟wall、8GiB额外内存；0 GPU、0真实模型forward/backward、0 optimizer updates、0新视频、0新增下载。已保存小checkpoint和配置可读。无需完整33B CPU加载。遇到协议冲突、超预算或需要GPU核实，只记录明确待测项并交付，不重复起草多轮方案。无AnyFlow/DMD/新memory/大训练，不改正式V3结果；Worker不操作Git。

## 验收

Judge审核真实模块参数计数、C1/C2数据/条件/梯度边界、CPU测试和可运行profile入口，并选择是否有限GPU放行。没有新生成视频是本轮CPU范围的正常结果。以能改变下一训练决策为收口标准，不要求解释所有失败机制。
