# Shift12 Step16：同模型的clean/generated history定位

最新：本组已完成，见[FINAL_RESULTS.md](FINAL_RESULTS.md)及完整诊断视频。以下保留启动和执行过程记录。

等待GPU1的shift12训练16次完成并通过初始化/采样审计后，只读其step16 checkpoint，在GPU3执行A/D39f、8 steps/chunk的clean-history诊断。其余与原GPU1 generated-history评测相同：原image/action/prompt/seed13/video-audio noise、RGB dual、CPU raw KV、action routing与feedback、推理shift2.22、native-FP32和完整stage1 bank。

每块commit及下一块RGB anchor使用原H3 teacher历史，而当前chunk仍由模型生成。必须与同checkpoint的generated-history结果一起看；该视频不能作为free-running或正式Demo验收。两条均24 noisy forwards+3 clean commits，完整39帧，不裁掉失败段。

不改模型或训练，不覆盖GPU1的自动生成队列。每条启动前重新检查GPU3空闲；如出现其它进程则报错退出，绝不中断其它任务。目录中的runtime/input hash冻结，checkpoint完成后单独记录四套权重hash。

训练已完成，GPU3诊断已实际启动；源run.json记录实时PID，归档为时间快照。teacher history本身含动作后的场景状态，clean视频光流正确不能单独证明当前chunk的action binding，不能与自生成历史的正式gate混用。
