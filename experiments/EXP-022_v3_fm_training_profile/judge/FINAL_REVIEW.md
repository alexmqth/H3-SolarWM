# EXP-022/v1 — Judge最终验收

> **验收更正：下文原工程PASS结论已部分撤回。当前Judge状态needs_revision，训练梯度协议不被接受；前向/KV、数据与执行事实保留。以[更正01](REVIEW_CORRECTION_01.md)及[当前判定](CURRENT_VERDICT.json)为准。**

2026-10-15 HKT。**accepted，工程训练通路PASS，任务收口。** 生成质量、动作、tail8容量、首块训练反向NOT_TESTED；不升级正式V3。

Judge审阅wrapper、profile runner、完整Worker报告与CPU终审，独立核对11个来源SHA、checkpoint0/step02 SHA、150项KV/RoPE精确一致、velocity/重载一致、11F/2B/2U事件计数及预算。见[独立审计](INDEPENDENT_AUDIT.json)。GPU无重复调用。

32train/8dev视频均已物化，其中36新视频2016帧已由Worker独立重解码；Judge审阅36场景总览，人物与环境可见，部分原始场景较暗。此轮不是逐帧人工评估所有源数据动作真值。36新MP4+contact合计62,967,508 bytes；原账本1,163,227,136为整文件系统free-space差值，有并发混杂，不能算本任务体积。

实际GPU进程413.200秒，11顶层DiT/2backward/2update；allocated20.293GiB、reserved22.852GiB。两次含GT clean commit更新29.243/26.062秒。事件采样最大RSS64.704GiB，结束58.848GiB，连续峰未测。checkpoint重算包含于backward耗时但精确次数未埋点；不因此补跑GPU。两个sigma不同的loss不表示收敛趋势。

训练仅当前C2，GT clean C1每次参数更新后重建；base/released Action LoRA冻结，新adapter梯度有限且有效变化。checkpoint重载恢复输出/RNG，不代表完整多步续训可复现已验证。首块12latent只测无梯度commit，不宣称其训练反向已测。profile参数1,720,320只是最小候选，不据此认定容量足够。

下一轮采用32条样本/128updates小数据拟合并配固定GT/self-generated-history验证，不采纳8/16/32每个短阶段都需先见画质收益才能继续的安排；这种门槛不足以判断训练效果，增加交接成本。维持协议/数值硬停，有限预算内连续执行。普通FM优先，AnyFlow/DMD仍不启动。具体授权见EXP-023。
