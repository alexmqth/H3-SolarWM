# 双局部指标诊断已完成

[完整结果、解释与图](RESULTS.md) · [分样本CSV](metrics.csv) · [噪声覆盖](NOISE_COVERAGE.md) · [136完整视频](../interval_consistency_candidate/FINAL_RESULTS.md)

128/control136/auxiliary136各A/D×3chunks×3噪声锚点，共54个case；每进程159 noisy+3 clean，无optimizer。参数/KV不变，128重复hash一致。finite/当前diagonal、finite到teacher插值或clean伪GT、到冻结128训练target分别报告，不能混称画质。

源目录：`H3-World/outputs/2026-10-08-16/stage1_dual_metric136/`。归档不含大target tensors和完整runtime，脚本需原目录执行。GPU4/5诊断已结束；全程与GPU6训练/评测合计最多3张卡。
