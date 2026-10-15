# EXP-024/v2：全层普通 FM C1/C2 有限 GPU profile

Judge，2026-10-15。Branch B，Parent为冻结V3/EXP-023修复入口。v1 CPU准备和原授权保留；本文件在GPU_AUTHORIZATION.json由Judge签发后可执行。Worker获准后立即执行，无需用户重复确认。

## 问题与冻结协议

唯一问题：全50层QKV rank8 adapter在真实H3上能否完成C1/C2的正确反向，实际成本能否支持下一小数据基线？10,752,000训练参数，新零增量初始化，不接续023的tail8权重。Original H3 + released Action LoRA冻结。相较023同时扩层与补C1训练，属于联合基线方案，不能作容量单因素结论。

保留Single I0/native time/Global RoPE/own-action/action feedback/current-prefix feedback/严格causal clean raw KV/12+5/SDPA/h3_fp32。复用023 SELECTION首条train及本地编码，不下载/重编码。sigma=.5、独立CPU noise RNG130024，模型seed13；AdamW lr1e-4、betas(.9,.95)、weight_decay .01。

## GPU前门槛

1. 保留v1源码/CPU报告历史；实现cache_signature结构检查并对CPU K/V/RoPE完整payload计算SHA256；保留原抽样签名的范围说明。完整摘要开销计入profile，不能追溯称023也做过全量摘要。
2. 每块在optimizer前验证50层forward/backward路由完整、冻结参数无梯度、新adapter梯度finite且总norm>0；C2另验证commit50层和cache签名不变。C1失败不得启动C2。失败保存trace和尝试/完成计数。
3. CPU注入缺层/冻结梯度/同容量KV内容变化等故障验证optimizer前硬停；记录新源码SHA和旧清单对应关系。Judge检查后签发GPU_AUTHORIZATION.json。

## 执行与硬预算

单张启动时确认空闲的NVIDIA L40，优先GPU0；不占用他人进程，最多1卡。先C1一次forward/backward/update，再用更新后的adapter clean commit GT C1，最后C2一次forward/backward/update。**最多3顶层DiT尝试、2 backward、2 optimizer updates、1800 GPU进程秒；0 VAE/text/视频。** checkpoint层重算单列，不算额外顶层DiT。禁止失败自动重试或转别的候选。

执行入口profile_gpu.py，GPU_AUTHORIZATION.json为新授权。环境CUDA_VISIBLE_DEVICES=<唯一空闲物理卡>、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=4，项目.venvs/h3world/bin/python。外层watchdog为`timeout --signal=TERM --kill-after=10s 1770s`，内层1800秒；watchdog前后保存命令、PID、时间、退出码和已有阶段结果。不得沿用提案1850秒外层超时。

allocated≥44GiB、RSS≥128GiB、非finite、协议/来源不符、缺失梯度/错误路由、OOM、时间预算耗尽或其他异常即停；只记录失败，不自动增加卡/降分辨率/改loss。日间项目≤3卡，夜间≤8卡，本任务都只占1空闲卡。

## 交付与验收

逐阶段同步耗时、allocated/reserved及各自峰值、RSS、CPU KV bytes；actual/attempted forward/backward/update计数，50层route trace、adapter逐层梯度/冻结梯度、输入与源码SHA、命令/日志/退出码，WORKER_REPORT及根report更新并消息通知Judge。两次更新仅profile，不要求视频，也不从两次loss推断学习效果。

若两块执行和梯度正确，Judge以实测资源决定是否另立32样本C1/C2小数据训练；建议估算32对更新+匹配评测的预算，不能自动延伸为64/128更新。当前不启动AnyFlow/DMD或新Memory，正式V3保持冻结。
