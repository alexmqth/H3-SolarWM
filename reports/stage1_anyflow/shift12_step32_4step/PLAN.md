# 补齐匹配FM32的AnyFlow 4-step评测

GPU4/reserve6，在GPU1 shift12总32次训练完整结束后，只读取其step32四套adapter，生成A/D39f、4 steps/chunk。原GPU1控制器继续原定8-step评测及最多64训练，互不写入同一输出目录。

这是补齐NFE对照，不是新训练变量：原GPU1 step32只评测8步，GPU3匹配FM32会评测4/8步。保持全覆盖rank8/full-history训练、RGB dual、CPU raw KV、causal action/feedback、seed13、native网格/推理shift2.22、同image/prompt/noise。实际12 noisy forwards + 3 clean commits。

启动前GPU4空闲；运行前再次检查显存，若被其它进程占用则退出，不终止任何进程。等待中的controller会保存实际源进程/训练状态；只有源训练完整结束、初始化审计通过、32次及参数策略满足协议时才读取checkpoint，并冻结其hash。每条完成后逐张量检查conditioning、完整帧数和raw指标，不自动验收视觉。

该评测仍使用冻结runtime，cached.json的anyflow_training_shift可能误显示2.22；真实训练shift12以setup/checkpoint config及training.json为准，不改写旧输出。没有改变架构、目标函数、LR或anchor。没有Stage2与124f扩展。
