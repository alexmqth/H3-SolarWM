# 当前任务：EXP-028 Action–Video LoRA角色梯度诊断

2026-10-17 HKT，Judge。用户要求继续推进。已批准原Worker立即实现、通过CPU门槛后在空闲卡上自动执行一次有限GPU诊断，不再等待用户重复确认。

[任务书](submission/experiments/EXP-028_action_video_role_probe/taskbook_v1.md) · [授权](submission/experiments/EXP-028_action_video_role_probe/AUTHORIZATION.json)。此前10DiT/8backward/1800秒草案由本任务替代；冻结旧证据不修改。

## 问题与简化

同两个真实ABot样本、相同当前latent/真实动作/噪声，比较Original-visible与正确V3的rank-8角色梯度。只测实际LoRA空间，消除完整权重梯度统计成本。两图比较含历史/prefix/拓扑差异，不作单边归因；负cos不是动作退化或双Adapter有效的证明。

## 执行与预算

- CPU实现≤60分钟、4workers/RSS32GiB；任务目录内修改，复用冻结代码、角色划分/混合精度/重算去重CPU测试。
- GPU：单L40（允许共享，首次要求实际free≥40GiB），≤2400进程秒、14DiT尝试（2commit+8主forward+4identity参考）、8backward；0optimizer更新、0视频/下载/编码解码。GPU allocated<40GiB，RSS<96GiB。
- 固定两片、层0/49、sigma .25/.75、同一checkpoint0的A，B为0。先验证首对输出与梯度加和，再继续剩余预算。
- 当前8卡皆有其他任务，禁止抢占。CPU PASS后允许限时后台等卡，候选0/1/6/7、两次free≥40960MiB检查，不要求无其他进程，2026-10-17 13:15 HKT后不再启动；轮询≤每60秒一次，失败不重试换卡。到期提交资源阻塞状态。
- 09:00后项目最多3卡，本任务只占1卡。Worker记录实际排队/运行/完成状态，并消息通知Judge；不操作Git。

## 交付和决策

CPU预检、固定输入源码SHA、实际命令、排队状态、GPU账本、各角色LoRA梯度/加和误差/QKV指标和报告。只为决定是否值得做一个行为相关的角色分离干预；不自动加训、扩层、追加视频。结果出来由Judge审核并push。

**显存准入补充01：** 完全空卡不是必要条件。EXP-027峰allocated24.45/reserved29.14GiB；本轮Original-visible反向峰尚未实测，首次暂要求实际free≥40GiB。符合余量可与现有进程共享，预算/截止不变。[补充](submission/experiments/EXP-028_action_video_role_probe/ADDENDUM_01_SHARED_GPU.md)。

**启动恢复02已批准：** GPU1已达到free≥40GiB。首次启动在import run失败，0DiT/0backward；保留attempt00。Worker修复真实入口导入和末次backward计数，Original-visible必须使用clean GT C1可微参考（禁止暗换same-sigma）；CPU通过后自动GPU1单次恢复，≤2396秒，累计≤2400秒，14/8预算不增加。[补充02](submission/experiments/EXP-028_action_video_role_probe/ADDENDUM_02_STARTUP_RECOVERY.md)。
