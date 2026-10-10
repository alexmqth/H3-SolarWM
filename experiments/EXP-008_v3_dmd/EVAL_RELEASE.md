# EXP-008/v2 Judge视频放行

训练已完成并经独立审核，最终cycle8固定。Judge独立CPU preflight通过，21项来源及native8step匹配，0GPU calls。**现在批准GPU0（启动前确认空闲）按prefill→AA/AD C2→AA/AD C3评估，最多35forward/4VAE/0update/.35GPUh、allocated≤44GiB、最晚09:00。** 只此checkpoint，不扫参数或重跑基线。

共同FM8 C1/首39 RGB，DMD自己prefill KV；C2同历史/动作/噪声，C3各自历史。FM8与AF3原片复用，三列视频注明实际cycle8。训练fake loss尖峰及后续较高水平作为风险如实记录，不能由surrogate降低声明改善。

先看AA/AD C2全部新增帧；若已经持续主体/场景崩溃，则停止C3、提交Judge，将当前配置作为负结果收口，不为了凑满预算生成无价值后续。普通瞬态形变/边界/ghosting仍按PARTIAL继续原计划。OOM/nonfinite/协议错误立即停，不自动重试。评估产物归H3-World/outputs/EXP-008_v3_dmd_eval，不覆盖既有证据。
