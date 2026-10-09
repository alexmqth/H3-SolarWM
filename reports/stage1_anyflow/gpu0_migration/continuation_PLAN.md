# GPU0 后续续训：提高模型驻留预算

等待当前GPU0冻结时间MLP16-update训练及全部评测完成、控制器退出。若任一完整A/D配置通过短片numeric gate，则先等待视觉审查；否则恢复Adam/RNG继续到32次更新，评测native8 A/D，仍失败才继续到64次并评测native4/8。

用户确认GPU0独占可用。后续任务使用CUDA_VISIBLE_DEVICES=0、ABOT_VRAM_RESERVE_GIB=6：L40可见显存约44.4GiB，模型驻留预算约38.4GiB。这个参数是offload watermark，实际PyTorch allocated/reserved峰值仍由运行记录确认，并不是显存使用量的保证值。

训练数据、目标、LR、chunk、anchor、动作、噪声及time冻结策略不变；续训保留Adam/RNG。GPU0前16次更新及其评测使用reserve20，后续改为reserve6，因此两段的耗时/峰值不能直接用于评价算法加速或显存收益。当前这一轮训练不中断。出现OOM或其他异常则停止并保留日志/checkpoint，按实际峰值调整运行预算。
