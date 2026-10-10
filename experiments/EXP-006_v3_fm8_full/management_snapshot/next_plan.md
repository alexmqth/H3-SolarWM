# EXP-007 / v2 — V3-AF真实模型初始化与单步训练核查

2026-10-11，Judge。**AF0 CPU已通过；本任务AF1-warmup GPU获准执行。** 用户夜间连续研究授权至09:00，正式V3 Baseline与SW-G可行性已验收。本任务与EXP-006普通FM8独立，可使用不同实际空闲GPU；不覆盖冻结参考，不加载旧AnyFlow训练产物。

## 问题与实现

确认新的target-time student能在原V3 causal/KV/current-prefix协议上初始化复现、得到有效梯度、执行一次真实finite-map更新并正确保存权重。该阶段是工程/训练可行性核查，不宣称画质改善。

使用本目录interval_student.py（AF0已测试）。Original H3 + released Action LoRA，h3_fp32边界、Single I0、native video1000sigma/audio1000、fixed_prefix_timesteps=False、own-action+feedback/current-prefix、Global、首12后5、sigma=r=0 clean commit。最后8层rank8 QKV增量+新target-time MLP，gate0.25；base与released LoRA冻结，target从实际加载time MLP克隆。trainable参数名单/hash完整记录，dropout=0。不改action mask/prefix/time条件来挽救失败。

训练source为冻结V3的30步生成C1首12与C2 A端点，只训练第二块；来源不是GT或独立泛化集。A/D后续训练与独立噪声评估再在下一阶段列出。历史cache必须由当前student自己的权重/sigma0/r0构建；本轮不从旧teacher raw KV直接跨权重沿用。

## 执行与上限

1. CPU重新运行本目录preflight；冻结runner/config/本任务书/source/input/checkpoint SHA。GPU入口核对实际空闲卡和budget。建议GPU1，EXP006 GPU0可并行。
2. 原V3未装adapter：C1 clean commit一次，C2固定sigma0.6同noisy state forward一次；保存参考velocity或摘要。
3. 安装零初始化QKV与新target MLP，重建student C1 clean KV一次；同C2 noisy state、r=sigma一次forward，与参考比较maxabs/relativeRMS，fp32输出allclose atol/rtol1e-5。记录bf16/SDPA环境，超阈值先报告不默改。
4. 可选一次同状态r=0.2前向，验证target非对角效果与finite输出，不据此判断视频能力。
5. 一次logical batch4（2 diagonal/FM +1 endpoint +1 general map），每样本3 detached+1 gradient forward，4次backward；batch内固定student历史cache。沿用经AF0验证的noise-clean场/有限差分和raw1000单位，独立CPU generator seed170007，AdamW lr1e-4/betas(.9,.95)/weight_decay.01，grad clip1。只有target/QKV可训练。
6. 保存初始化与step1配套target/QKV/optimizer/RNG及协议metadata；更新后再重建一次C1 KV，证明cache随student变化且是detached。不得把该KV与旧权重cache混称。

**总上限22完整forward：初始化4 +可选非对角1 +训练16 +更新后prefill1。4backward、1 optimizer update、0VAE、≤0.75 GPU小时，含加载/保存/失败；CPU4线程，allocated peak≤44GiB，不reset peak。** 不必另做训练前prefill，复用步骤3在本次optimizer更新前的同权重student cache即可。每次forward/backward/update预记账；绝对deadline min(阶段预算,09:00)，单进程锁。无自动重试，不在此阶段自行继续step2。

## 验收和停止

初始化对角回归、合法时间行、真正有限差分训练calls、finite梯度/参数更新、冻结权重不变、只读cache及checkpoint反向不变、更新后自建KV变化、实际显存/耗时/完整保存可追溯。检查target/QKV参数各有有效更新，但不要求零初始化LoRA的每个矩阵在第一步都发生变化。

OOM/nonfinite/协议错误/上限立即停止；允许提交负结果，Judge选择一次明确修复或停止，不无限调整。stdout/log/config/checkpoint元数据/机器结果及root report task状态同步。新checkpoint在输出目录，Git只存manifest/代码/小日志，不复制base模型或每步缓存（当前磁盘剩余约392GB）。

完成释放GPU，Judge按真实单步成本决定后续最多32updates及匹配8NFE评估。普通FM8与AnyFlow训练区别必须清楚；此任务不含DMD、额外adapter或全历史反传。


## 前置任务EXP-006已收口

全程普通FM8 AA/AD73已通过Judge有限可行性验收，quality PARTIAL。实际43forward/5VAE/492.856秒；首窗不再借用30步。证据在submission/experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md。当前GPU任务为本EXP007初始化+1update；32updates/DMD尚未自动放行。夜间连续监督目标保持到09:00，期间按各阶段结果布置下一步，09:00后最多3卡。
