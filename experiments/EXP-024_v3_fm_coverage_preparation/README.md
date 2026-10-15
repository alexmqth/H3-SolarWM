# EXP-024/v2 · V3 普通 FM 训练覆盖与全层 GPU profile

CPU 阶段与Judge授权的单卡两更新profile完成，待最终验收。已审计EXP-023 checkpoint0/128、实现C1/C2训练入口、检查34条真实编码条件与50层微型checkpoint backward生命周期。真实33B full50 rank8在GPU0完成C1/C2各1更新；**没有生成视频或运行32样本训练**，不代表质量提升。

[Worker 报告](WORKER_REPORT.md) · [GPU实测结果](GPU_PROFILE_RESULT.json) · [启动记录](GPU_PROFILE_LAUNCH.json) · [故障注入](GATE_FAULT_CPU.json) · [参数审计](PARAMETER_AUDIT.json) · [CPU 小张量测试](CPU_TEST_RESULTS.json) · [50层生命周期](FULL_LAYER_CPU_RESULTS.json) · [真实条件布局](REAL_INPUT_AUDIT.json) · [v2任务书](taskbook_v2_gpu_profile.md)。

Judge已accepted工程profile，视频能力未测。[最终审核](judge/FINAL_REVIEW.md) · [独立复核](judge/FINAL_AUDIT.json)。下一EXP-025全层C1/C2小数据拟合已单独授权。
