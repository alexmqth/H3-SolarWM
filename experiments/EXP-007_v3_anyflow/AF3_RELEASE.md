# EXP-007/v3 AF3独立视频预算放行

## Judge阶段放行 — 2026-10-11 03:21 HKT

AF2已完成并接受工程结果：累计step32，新增31updates/527forward/124backward/0VAE/4287.86071秒，peak26.76744GiB，进程已退出。配对checkpoint及完整账本独立审计通过，见submission/experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md。

**现在批准执行AF3全部预定义视频阶段**，使用AF2最终step32：shared C1 prefill→AA/AD C2→AA/AD C3，共35forward=1prefill+2commit+32sampling、4VAE、0update、≤0.35GPU小时、allocated≤44GiB、最晚09:00。建议实际空闲GPU0；不覆盖原片。CPU preflight先冻结源码/config/输入/配对checkpoint，通过后直接启动，无需用户再次批准。准许marker为实验目录AF3_GPU_AUTHORIZATION.json。异常停止交Judge，无自动重试，不扩到其他checkpoint或NFE。

C1为EXP-006 FM8共同clean历史与39RGB；AF用自身权重prefill KV。C2同可见历史比较、C3各自生成历史。保存完整AA/AD73原片与匹配对比、逐块耗时/显存/缓存与RGB不变性。训练C1来自原30步V3，评估C1来自FM8，这一历史分布差别如实记录。训练结果不直接构成生成能力验收。

## Judge修复重试放行 — 2026-10-11 03:26 HKT

AF3首次prefill因缺少冻结runtime/code/causal导入路径，在模型forward前退出：0forward/0VAE，3.560679秒已记入原账本。修复路径、补充CPU真实导入预检及依赖摘要后，Judge独立CPU_PASS（step32、18项source、native8step）。批准一次修复后attempt2，沿用同输出和累计账本；原35forward/4VAE/0.35GPU小时/44GiB及09:00截止不变。保留attempt1日志与manifest，后续异常仍停止交Judge。
