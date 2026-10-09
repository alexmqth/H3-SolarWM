最新：128训练与A/D39f全部4/8步评测已完成，队列正常结束。8步A+0.040414/D−0.719245/分离度0.759660；4步分离度0.134144。画质仍失败，见[STEP128_RESULTS.md](STEP128_RESULTS.md)。[同模型历史对照](../history128/FINAL_RESULTS.md)与[固定历史动作干预](../counterfactual128/FINAL_RESULTS.md)也完成。没有自动追加训练。下文保留历史快照。

13:45核对：真实控制器/torchrun均存活，已完成86次更新，step80已保存。仍在训练96，尚无96视频。新增[64次分时段诊断](../duration_response/RESULTS.md)。

# 已验证step68的四卡续训：GPU4–7，正在运行

2026-10-08 13:29:21启动，控制器2446816、torchrun2447766，源目录outputs/2026-10-08-13/stage1_parallel_resume68_to128/。输入988项冻结，重用上一轮完全相同的冻结runtime，仅改physicalGPU与续训入口，从已成功训练的step68继续96，不重跑64→68。训练状态running、resume68；已核查实际GPU保存的预更新checkpoint，四套adapter、Adam、teacher/update history、固定配置与三类RNG完全等于原step68。目前没有96视频结果。

上一轮[64→68成功记录](../parallel_duration128/TRAIN68_RESULTS.md)包含实际Adam/RNG恢复和四replica一致性。原控制器在96启动前因GPU3新进程占用而被idle guard拒绝；既有训练成功与后续硬件拒绝分别保留，没有修改旧run.json掩盖停止。GPU0同样被其它任务使用，因此本队列使用启动前实测空闲的4–7；GPU0不限25GiB授权和reserve6配置保留。

- [有限训练预算/固定协议](PLAN.md)
- [真实启动记录](launch.json)
- [源64完整效果与可播放视频](../shift12_duration64/FINAL_RESULTS.md)

到96评测A/D8；数字gate通过则补4并停待完整视觉、匹配FM、独立seed及动作切换检查；否则再到最多128评测4/8并停。仍要求A>0、D<0、A−D>1且人物/场景完整。Stage1尚未验收，Stage2暂缓。报告内控制器是审阅用源码，依赖工作目录中的冻结runtime，不是从报告目录直接启动的训练包。
