# EXP-026/v3 Judge 审核

2026-10-16 HKT。接受一次失败尝试的证据交付，Worker执行failed；失败原因是FlashAttention/PyTorch ABI不兼容，四卡模型显存可行性未得到回答。不能归因于四卡容量不足。

独立核对四rank日志及源SHA：均到SP4/DP1、shard4/replica1 NCCL ready，随后在set_attention_backend(flash)报错并释放。stderr含flash_attn_2_cuda.so undefined C++ symbol。26.316秒、105.265GPU秒，一次尝试低于2400GPU秒预算；0forward/codec/训练。LoRA/FSDP/EMA未到，14/14读取进度不代替strict receipt。实测物理卡0/1/6/7已归零。allocated为到达检查点的0，不据此推断完整模型峰值。

CPU准备存在约4秒12rank并行超出8并发；已结束并记录，v2八rank JSON与Git原件逐字节一致。接受该已纠正执行偏差，不为此重复任何研究实验。后续必须串行组织CPU组测试。

环境版本torch2.6.0+cu124、flash-attn2.8.3；Worker CPU import重现ABI错误，既有缓存同一坏binary，h3world torch2.10不能直接替换。此依赖门控应在GPU前完成，下一轮必须补上。

决定：只做一次有限的隔离FlashAttention修复，保持torch/模型/attention协议不变；CPU真实import通过后，授权v4一次同范围load-only重新尝试。v3日志不覆盖，105.265GPU秒继续计入累计成本。官方数据访问仍独立阻塞，视频能力NOT_TESTED。
