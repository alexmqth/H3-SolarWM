# EXP-026/v2：L40内存安全加载的CPU修复准备

2026-10-16 HKT。v1输入访问BLOCKED_ACCESS，GPU未运行；证据在v1_frozen及judge/V1_REVIEW.md。现在授权原Worker立即执行独立的加载路径CPU修复，数据权限等待用户，不重试401/403。

## 单一问题

官方33B在FSDP前整体.to(cuda)超过L40容量。为同一官方Stage2-fix推理准备CPU加载后分片的最小兼容修复，保持权重、dtype、attention、相机、RoPE、KV、sampling与输出协议不变。不能把CPU模拟称为真实FSDP/GPU通过。

## 实际交付

1. 读取load_transformer、run_inference、wrap_h3_fsdp、LoRA注入及EMA loading路径。先判断显式受支持CPU device_map是否足够；如需源码改动，在本实验memory_load_v2/保存最小patch与base revision/SHA，不直接改冻结SolarWM checkout及已有V3。
2. 修复必须避免FSDP前whole-model CUDA搬运，并确认逐block包装顺序、根模块残余、FP32单元、ignored LoRA参数、EMA加载不会重新聚集完整模型到单卡。明确每rank CPU副本/瞬时峰及内存释放。不要顺带引入量化、改NFE、降分辨率或改推理语义。
3. 用小模型/桩执行真实生产边界的CPU测试：加载设备与dtype、LoRA/EMA数值保存、调用顺序；故障注入整模型.to(cuda)必须触发拒绝。测试可在临时副本应用patch并确认语法/官方config可解析；不能伪造真实分布式cuda/FSDP结果。
4. 记录尚需GPU验证的风险，提交可review的具体patch、CPU结果、来源清单及下一阶段有限load-only GPU建议（几卡、总GPU秒、峰值/退出条件，区别加载验证和视频生成）。当前仍有数据缺失，不给空白sample_id配置标PASS。

## 预算及停止

0GPU初始化/forward/encode/decode/updates，0新网络下载；CPU≤8并发，RSS≤32GiB，新磁盘≤2GiB，45分钟。无需把33B完整权重载入CPU来证明代码顺序；使用小模型即可。复杂改动若需重写FSDP/改权重语义，停止并说明，不为了跑通不断扩大实现。完成即主动消息Judge，不等用户催，不操作Git。

数据、完整reader和实际GPU仍分别blocked/not tested。本版本只批准可独立开展的CPU实现工作；未来GPU必须另发具体授权，夜间≤8空闲卡、09:00后≤3卡上限照旧。
