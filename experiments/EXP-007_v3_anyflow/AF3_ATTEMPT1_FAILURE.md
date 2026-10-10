# AF3 prefill attempt 1 — stopped before model load

2026-10-11 HKT。正式 AF3 已由 Judge 授权；首次 GPU0 入口命令 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-007_v3_anyflow/run_af3.py --stage prefill --gpu 0 > H3-World/outputs/EXP-007_af3_prefill.log 2>&1` 在 `from benchmark import write_video` 处退出。原因是 AF3 runner 的 `sys.path` 漏掉冻结 runtime 的 `code/causal` 目录。发生在加载 33B、执行 prefill 或创建 chunk row 之前。

原始 `H3-World/outputs/EXP-007_v3_anyflow_af3/budget.json` 和 stdout 保留：**0 forward、0 sampling、0 commit、0 VAE、3.560679436 秒 GPU0 预留时间**。原始源码清单复制为 [attempt1 manifest](source_manifest_v3_af3_attempt1.json)。没有任何 cache、latent、RGB 或视频产物。

Worker 已在独立代码中补上路径，并让 CPU preflight 实际导入 `benchmark`、action flow 与 FM8 helper；更新的 source manifest 核对 18 项来源，CPU 预检 PASS、0 GPU calls。根据 taskbook“异常停止交 Judge、无自动重试”，**此后尚未再启动 GPU**。如 Judge 批准修复重跑，应保留原账本和日志，在同一累计账本中从总 0.35 GPUh 扣除 3.560679436 秒，并把第二次 stdout 保存为不同文件；不得重置计数。
