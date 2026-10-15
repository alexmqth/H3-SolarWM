# 2026-10-16 Judge更正：8卡不是固定158帧推理的最低要求

用户询问为何必须8卡。独立源码核对：SolarWM config.py::_validate_inference_distributed对fixed Stage2要求SP4且world_size可被4整除，并不要求world_size=8。source-length模式另有8卡SP8约束，与本任务fixed158帧不同。

Judge从v2候选在内存中仅改world_size=4、保留SP4、sample_count=1、balance_by_dataset=false，调用官方validate_h3_config成功。未写运行配置、加载模型、初始化GPU或启动实验。

此前8卡要求来自官方示例world8/SP4/DP2和本轮补丁对world8/shard8/两样本的硬编码，不是已证明的模型最低硬件要求。默认按DP分片在四卡SP4/DP1下仍不可行，需将冻结base显式分片到物理四卡。现有frozen_base_groups要求shard_size匹配LOCAL_WORLD_SIZE，其通用代码未限定必须8；四卡真实FSDP与峰值显存尚未验证。

仅权重算术：61.728GiB base分4卡约15.432GiB/卡，加复制BF16 LoRA约3.866GiB/卡，总约19.298GiB/卡。尚不含all-gather、激活、KV、codec、临时buffer，不能当峰值或fit保证。四份base+LoRA同时常驻CPU约262.377GiB；旧524.753GiB/650GiB提案属于八副本方案，不适用于所有候选。

下一决策优先考虑四卡SP4/DP1单样本的最小调整，而非继续等待八卡；本更正未下发新任务或GPU预算。官方数据访问阻塞不变。最终审核的历史CPU通过范围保持，原“需八卡”应理解为v2补丁限制。
