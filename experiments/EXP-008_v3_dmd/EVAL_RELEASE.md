# EXP-008/v2 Judge视频放行

训练已完成并经独立审核，最终cycle8固定。Judge独立CPU preflight通过，21项来源及native8step匹配，0GPU calls。**现在批准GPU0（启动前确认空闲）按prefill→AA/AD C2→AA/AD C3评估，最多35forward/4VAE/0update/.35GPUh、allocated≤44GiB、最晚09:00。** 只此checkpoint，不扫参数或重跑基线。

共同FM8 C1/首39 RGB，DMD自己prefill KV；C2同历史/动作/噪声，C3各自历史。FM8与AF3原片复用，三列视频注明实际cycle8。训练fake loss尖峰及后续较高水平作为风险如实记录，不能由surrogate降低声明改善。

先看AA/AD C2全部新增帧；若已经持续主体/场景崩溃，则停止C3、提交Judge，将当前配置作为负结果收口，不为了凑满预算生成无价值后续。普通瞬态形变/边界/ghosting仍按PARTIAL继续原计划。OOM/nonfinite/协议错误立即停，不自动重试。评估产物归H3-World/outputs/EXP-008_v3_dmd_eval，不覆盖既有证据。

## Judge提前停止 — 最终cycle8出现持续崩溃

AA C2全部17张新增帧为彩色噪声，主体与场景不可辨。停止本配置的后续推理与训练，C3 AA/AD取消。发现时AD C2进程1695658已经启动，允许其完成原预算内当前块，之后无新GPU任务；当前marker已撤销新启动资格，原授权保留EVAL_GPU_AUTHORIZATION_INITIAL.json。按实际完成的56帧证据收口，不把未测C3记为完成，不换checkpoint挑片或加训练挽救。
