# Full-history AnyFlow有限训练量对照

step32训练与A/D8-step评测均已结束，A=−1.309516、D=−1.497935、分离度0.188419，未通过动作门槛，画面相对16次无明确改善。[完整结果和视频](STEP32_RESULTS.md)、[训练曲线](GPU_RESULTS.md)。GPU0已从step32精确续到总64，reserve6、不限25GiB；[执行计划](PLAN.md)、[队列快照](run.json)。

源目录：H3-World/outputs/2026-10-08-10/stage1_full_history_duration64/。本目录仅保存小型证据；run_extension.py是依赖该机器冻结runtime的控制器归档，迁移机器请使用REPRODUCE.md的通用trainer/benchmark入口。独立GPU仍有微小反传重复性误差；恢复state一致不保证CUDA优化轨迹逐bit重放。

真实GPU初始化后保存的step16已逐项核验：四套adapter张量、Adam、更新历史、teacher身份及CPU/CUDA/logical RNG与来源全部完全相同，见[gpu_resume_audit.json](gpu_resume_audit.json)。

step32→64的实际GPU回载也已逐项核验：四套adapter、Adam、更新记录、teacher身份与三种RNG全等，见[gpu_resume32_audit.json](gpu_resume32_audit.json)。
