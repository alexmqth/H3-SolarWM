# EXP-023 第一次训练启动失败（梯度门控前）

`eval_pre_retry2` 已提供 6 条可播放的 checkpoint0 视频及固定 σ probe。本尝试加载相同 `checkpoint_000.pt`，第一步完成 GT C1 clean commit，随后在进入 C2 的 `current_prefix_feedback()` 上下文时触发 `verify_one_edge_change`：埋点先于上下文安装，使源码保护看到的是审计 wrapper，而不是原始 `current_prefix_attend`。

账本状态保持 `FAILED_STOPPED`，耗时 26.962221870955545 GPU 秒、2 次顶层 DiT **预留**（其中 1 次 C1 commit 实际完成，C2 在函数入口失败，未执行模型前向）、0 次 backward、0 次 optimizer update。原 `checkpoint_000.pt` 未修改，未生成新 adapter checkpoint。失败账本、日志、当时源码及 manifest 在本目录逐字节备份，上级原文件继续保留。

修复仅改变埋点生命周期：生产 `train_one` 使用 `verified_attention_scope`，先进入并完成路由源码验证，随后安装审计 wrapper，覆盖完整 forward 和 checkpoint backward，并在退出上下文前恢复。CPU 使用实际 helper 与小张量真实 `ChunkAttention` 验证：有无审计 hook 的输出和梯度一致，连续两次及正/反向异常后全局函数均恢复。见 `../HOOK_LIFECYCLE_CPU_V2.json`。下一次 GPU 训练恢复必须累计本次失败费用，且首步真实反向仍须在 optimizer 前通过。
