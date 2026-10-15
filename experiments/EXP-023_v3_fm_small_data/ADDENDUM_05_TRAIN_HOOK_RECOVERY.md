# EXP-023/v1 补充05：修复审计hook安装顺序后恢复训练

2026-10-15 HKT，Judge已独立复核CPU生产helper，批准立即只恢复train→post。既有输入、媒体修复和梯度协议门控继续有效，不扩大预算。

## 故障与已完成修复

原train_one在进入current_prefix_feedback之前替换current_prefix_attend为audited，入口源码检查因此拒绝执行。训练attempt耗时26.962221871GPU秒，预记2DiT（实际仅C1 commit执行，C2在context入口失败），0backward/0update。保留原TRAIN_BUDGET的FAILED_STOPPED及attempt_02代码/manifest/日志；计数按保守2次reserve，不回退额度。

生产train_one现在复用fm_run.verified_attention_scope：先进入原context完成验证，再安装hook，保持至forward和backward结束，finally在退出context前恢复模块函数。Worker HOOK_LIFECYCLE_CPU_V2与Judge judge/HOOK_SCOPE_CPU_REVIEW均直接调用该生产helper，不另写类似生命周期。真实小张量attention checkpoint on/off、有/无hook梯度一致；连续调用两次一致；forward/backward异常后两个全局函数均恢复。CPU报告不代替真实33B首步门控。

## 立即执行

- 冻结本修复的新来源revision与补充授权，保留失败revision。不改变loss、输入、mask、采样和训练参数，仅修hook生命周期和累计恢复入口。
- **不重跑已经完成的6条pre-eval视频、固定probe、34clip编码或checkpoint0入口回归。** 验证已保存pre产物身份/manifest，直接从未改变的checkpoint000开始原seed/数据顺序的128updates。
- 首次真实backward继续按补充02记录实际重算路由/层、cache不变和梯度，仅通过后才能optimizer.step；通过即自动继续剩余更新与post6片/固定probe，无额外marker。
- 恢复train使用独立attempt账本；原train失败26.962秒与2reserve和两次媒体失败均累计，不伪造为完成、不覆盖证据。已完成eval_pre_retry2可作为前置条件，不受失败attempt状态错误阻塞，但失败成本始终计入。
- 模型/协议/梯度/OOM/预算错误硬停；已持久化RGB的媒体故障只CPU恢复，不重复模型采样。

## 预算

**全部上限保持：449顶层DiT、128backward/updates、70VAE encode、14decode、600text、13800GPU秒（编码4200/训练6000/评测3600）**，最多2空闲卡/训练1卡及原host RAM/磁盘限制。训练6000秒包含原失败attempt。当前累计2910.864GPU秒、106DiT reserve、8decode、70encode、151text、0backward/update；计划剩余256训练forward+82post约338，预计累计444，低于449；实际事件账本为准。

Worker收到即恢复，无需等待用户或新授权。首步门控通过、32/64/128updates、完成/异常及时消息通知Judge。工程协议通过与生成能力结论继续分开。
