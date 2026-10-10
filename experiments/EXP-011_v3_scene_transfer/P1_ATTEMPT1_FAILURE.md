# P1 attempt 1：模型加载前失败

2026-10-11 05:53 HKT，在 Judge 第一版 P1 marker 下启动 GPU0 编码入口。进程在建立 `P1_result.json` 之前，因为 `sha(__file__)` 将字符串传给仅接受 `Path` 的函数而抛出 `AttributeError`。完整 traceback 保存在 `H3-World/outputs/EXP-011_v3_scene_transfer/P1.log`，机器可读事件在同目录 `P1_attempt1_failure.json`。

**没有加载文本编码器、VAE 或 DiT；没有产生 fixture。** 原始账本 `budget_attempt1_raw.json` 显示 0 次模型调用，仅有 stage_start。为保守计入失败尝试，当前 `budget.json` 增补了明确标注的 stage_stop，并按整个 shell 命令耗时计 4.386366 GPU 秒；这不是实际计算内核用时。原始账本保留不覆盖。

已在 CPU 上将 `common.sha` 改为同时接受 `Path` 和字符串，并把 P1 行初始化纳入异常捕获区；`test_p0.py` 增加字符串路径回归检查。修复后的代码清单重新冻结，原 marker 的代码哈希将失效。**未自动重试**；需 Judge 复核并发新的 P1 marker 后才可再次启动。

`attempt1_archive/` 保存了首发 marker、原始日志/账本和失败事件。根据修复前后的精确补丁重建的 `common_before_fix.py`、`encode_native_attempt1.py`、`test_p0_attempt1.py` 与 `code_manifest_attempt1.json` 已逐字节核对：重建 manifest SHA-256 为 `950bde5ce11460a8bbf183a74829302ced44f5b135fe891302178d27b253b4a4`，与首发 marker 中的代码清单哈希一致。当前修复后 manifest SHA-256 为 `11363412837dfdca85a611761fc1a59b730cb2c541117942fbbcf14f24866976`。
