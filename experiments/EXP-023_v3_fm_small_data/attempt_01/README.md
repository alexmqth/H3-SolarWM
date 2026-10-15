# EXP-023 checkpoint0 评测首次恢复（失败记录）

本目录保留补充03下 `eval_pre_retry` 的 runner、恢复来源清单、日志和账本快照。此次复用原始 `checkpoint_000.pt`，第一条 `train_constant` GT-history 再次完成 1 次 C1 commit、8 次 C2 FM 采样和 1 次 VAE decode。随后新增的媒体身份记录调用了 bfloat16 噪声张量的 `.numpy()`，抛出 `TypeError: Got unsupported ScalarType BFloat16`；错误发生在 RGB 原子落盘之前，因此仍无可供 CPU 恢复的 RGB、latent 或有效 MP4。

此次失败独立消耗 81.93666345003294 GPU 秒、9 次顶层 DiT、1 次 VAE decode、0 次反向与更新。原始 `EVAL_PRE_RETRY_BUDGET.json` 在上级目录仍保持 `FAILED_STOPPED`；本目录为逐字节备份。与首次失败合计，评测已消耗约 383.001 GPU 秒、22 次 DiT 和 2 次 decode；后续以两个原始账本的精确数值相加。

修复后先持久化 decoded RGB，再执行身份计算和编码。真实 bfloat16 条件的 GT/self 两条写出路径、并排片以及身份/codec 故障注入均已通过 CPU 测试，见 `../MEDIA_SMOKE_V3.json`。下一次 GPU 恢复必须有新的 Judge 授权，且两次失败成本都要计入总账。
