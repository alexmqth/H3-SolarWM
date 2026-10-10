# EXP-008/v1 Judge GPU放行

AF3有限续写可行性及全部独立审计已通过，质量PARTIAL，无已证实FM8收益。DMD迁移入口经过Judge差异审查及独立CPU preflight：23项冻结来源、AF2 step32配对权重、正式任务书、实际causal.dmd导入与SHA通过，0GPU calls。tiny-H3完整8map链CPU结果通过，不冒充真实33B结果。

**批准一个DMD pilot cycle，GPU0/2/5（当前均空闲），17forward/3backward/3update/0VAE，≤30min wall、保守3×wall≤1.5GPUh，每卡allocated≤44GiB，最晚09:00 HKT。** teacher/fake/student独立设备与KV；fake两次FM更新、student一次完整8map DMD更新。若失败立即停止交Judge，无自动重试，无第二cycle授权。已有V3和AF结果保持冻结。

授权marker在dmd_cpu/DMD_GPU_AUTHORIZATION.json。源码/config/taskbook已冻结，运行后不能静默改协议。结果和日志归入H3-World/outputs/EXP-008_v3_dmd_pilot，Worker完成后报告，由Judge根据真实多卡成本与梯度正确性决定后续。
