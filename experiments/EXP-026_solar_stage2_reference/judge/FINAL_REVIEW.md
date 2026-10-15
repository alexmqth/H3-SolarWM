# EXP-026/v2 Judge 最终审核

2026-10-16 HKT。**accepted：有限CPU补丁准备与阻塞证据。官方数据/reader及实际GPU推理仍BLOCKED/NOT_TESTED，未完成官方复现。** 本轮0GPU/0新下载/0updates，无官方输出视频。Worker停止本轮实现，等待新授权。

## v1结论更正

v1只有通用YAML loader通过；H3 backend validate_h3_config拒绝SP8。SP8/DP1默认FSDP也不分片模型。Judge此次独立复现拒绝结果，v1审计的generic loader事实保留，不能读作backend通过。v2改为官方SP4/DP2、两样本，backend检查通过；样本数仍只是候选配置，无真实样本。

## 补丁验收范围

独立核对两个官方源码base SHA、patch在临时副本应用及modified SHA/语法，验证v2 H3 backend SP4。源码检查确认：opt-in CPU加载；只搬ignored LoRA；调用现有frozen_base_shard_size=8；未改官方checkout、时间/相机/RoPE/KV/采样协议。默认旧分支保留。

Worker小模型测试验证CPU dtype/device、旧whole-model CUDA搬运故障注入、load→inject→wrap、tiny EMA恢复0.75；八进程Gloo记录每rank shard8/replica1。Judge检查测试实现与记录，未重复Gloo。包装器被桩替换的测试不证明真实FSDP模型工作；真实33B/4GB EMA/CUDA混合设备、跨SP collectives和瞬时显存均NOT_TESTED。[独立审计](V2_AUDIT.json)。

## 成本与风险判断

61.728GiB base加约3.866GiB LoRA，八份同时常驻的简单容量估算524.753GiB。Worker观察MemAvailable546.531GiB、GPU2–5占用。该估算不是已测峰，也不是所有加载方案的不可降低内存下界；共享文件页、mmap及迁移时序可能改变实际RSS。本轮没有验证这些优化。650GiB门槛只是该加载提案的保守条件，尚未授权为可运行任务。

单进程CPU测试峰RSS0.891GiB；8进程Gloo聚合RSS未采样，不能宣称聚合峰已测。未为该低成本准备重复额外实验。

## 决策与恢复条件

接受有限交付，停止追加加载工程。本轮GPU提案8×600秒只是请求，**未获批准**；不创建空闲卡轮询队列，不动他人进程。

1. 用户取得官方数据访问并在qma账号登录，或提供本地数据路径；实际读取index/support/payload、锁定两样本后才可验收reader。当前HF401/ModelScope403仍来自Worker记录，Judge未重复下载。
2. 在允许的夜间窗口具有足够空闲GPU和CPU内存时，Judge重新审核一次具体load-only入口与预算；数据暂缺不影响理论上独立设计加载检查，但当前资源/收益不支持直接启动。
3. 真正加载通过且官方输入可用后，另定两条158帧官方视频任务；不宣称当前已实现标准推理或可比较生成能力。

[完整CPU交付](../memory_load_v2/WORKER_REPORT.md) · [补丁](../memory_load_v2/cpu_sharded_load.patch) · [未来加载提案](../memory_load_v2/GPU_LOAD_ONLY_PROPOSAL.md)。数据权限和资源是当前外部条件；不继续无上限重写FSDP。
