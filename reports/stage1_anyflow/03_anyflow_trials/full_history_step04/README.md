# Full-history AnyFlow step04 A/D8诊断

GPU1完成无历史重复梯度探针后，评估GPU0主训练已保存的step04。初始化/推理协议与step16相同，仅checkpoint步数不同。完整39f、8 steps/chunk、generated history，预期24 noisy+3 commits；必须保存conditioning逐项相等。与GPU0训练并行，单次时间不作公平速度比较。不替换正式meeting视频。
