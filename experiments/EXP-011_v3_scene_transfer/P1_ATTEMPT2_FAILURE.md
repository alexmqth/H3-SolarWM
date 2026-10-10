# P1 attempt 2：离线模型目录未注入

2026-10-11 05:56 HKT，Judge 授权的一次修复重跑在 GPU0 启动。`load_encoding_pipeline()` 没有经过冻结 `infer.py` 的环境设置；独立进程里的 `DIFFSYNTH_MODEL_BASE_PATH` 因而为空，DiffSynth 的 `ModelConfig` 在错误工作目录找到 `[]`，于模型类型识别阶段退出。完整 traceback、失败 `P1_result`、原始账本、当时的冻结 `common.py`/`encode_native.py`/`code_manifest.json` 与 attempt2 marker 都保存在 `attempt2_archive/`；输出目录中另存 `P1_attempt2.log` 和 `P1_result_attempt2.json`。

这次也没有加载 H3 权重、文本编码器或 VAE，更没有产生 fixture；账本模型调用计数仍全部为 0。Runner 的 active ledger 实记约 0.001495 秒；为与首次失败一致，保守按整个 shell 命令 4.280806 秒计入 P1 总预算。两次尝试累计保守计时 **8.667172 秒 / 900 秒**；原始账本快照保留。

Judge 指示先用 CPU 验证在命令环境中显式设置 `DIFFSYNTH_MODEL_BASE_PATH` 后能否解析冻结 runtime 的 text encoder、source VAE 与 processor。**当前冻结代码已恢复至 attempt2 的 manifest SHA `11363412837dfdca85a611761fc1a59b730cb2c541117942fbbcf14f24866976`，不修改模型协议。没有自动启动第三次 GPU 尝试；新的命令环境仍需 Judge 单独放行。**
