# EXP-023 checkpoint0 评测第一次尝试（失败记录）

本目录保留 2026-10-15 首次 `eval_pre` 的冻结 runner、输入 manifest、日志、账本和成功的固定入口回归。该次运行在第一条 `train_constant` GT-history 视频完成 9 次 DiT 调用和 1 次 VAE 解码后，因系统中不存在命令名 `ffmpeg`，在保存 MP4 时停止。没有 MP4、已解码 RGB 或生成 latent 落盘；不可把此尝试记为有效视频评测。

原 `EVAL_PRE_BUDGET.json` 仍在上级目录，状态保持 `FAILED_STOPPED`；此处为逐字节备份。失败耗时 301.0642068160232 GPU 秒，13 次顶层 DiT 前向（4 次固定入口回归 + 9 次首片），1 次解码，0 次反向和参数更新。固定入口回归的 velocity 误差为 0，150 项 KV/RoPE 精确一致。上级 `checkpoint_000.pt` 保持原 SHA，不重新初始化覆盖。

受限恢复见 `../ADDENDUM_03_VIDEO_IO_RECOVERY.md` 与 `../resume_run.py`。其账本必须累计本次成本。
