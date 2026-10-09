# Stage1 精度诊断：同状态计算，尚非画质修复

官方SolarWM固定版本ce1da4e的optional.py明确保留六组FP32层：video/audio输入输出投影，以及时间MLP两个线性层。官方调制层先对FP32时间嵌入做SiLU，再转为BF16进入AdaLN投影。本地历史路径对全部base权重/时间嵌入使用BF16，AnyFlow混合后也转为BF16。

此前真实时间MLP的CPU有限差分审计，在相同已舍入权重下发现embedding导数存在明显量化误差；这不证明完整velocity或视频失败由它导致。因此先进行受控诊断。

- 等待GPU0的frozen16六条视频评测完成且控制器退出后，单独使用GPU0，reserve=6GiB。
- 加载同一frozen16 checkpoint，不做任何optimizer更新。
- A/D，chunk0/2，每个chunk使用固定验证noise和两个时间对（endpoint/general-map）。
- 三种计算：legacy、time_fp32、boundary_fp32。后者包括时间及video/audio输入输出投影。
- 各种计算共享legacy生成的clean CPU KV、noisy latent以及有限差分方向，避免将历史变化混入当前score比较。
- 只改变计算dtype，权重数值始终是原本已BF16舍入的值；没有重新加载checkpoint中原生FP32权重。latent进入模型时的BF16转换也保持原状。因此不是完整的官方精度策略复刻。
- 记录raw residual、velocity相对差、有限差分向量相对差/cosine、时间和实际显存峰值。没有“正确导数”的完整33B参考，不把差异当成误差下降。
- 每个样本结束后恢复原精度，要求输出逐bit复现；所有hook只存在于本进程，源代码及运行中实验不变。

cpu_validation.json使用真实小型H3和DiffSynth offload wrapper检查三种profile的cache只读、checkpoint反传、非零有限梯度、dtype/offload切换及逐bit回滚，已通过。33B结果以run.json为准；CPU通过不等于GPU或画质通过。

run_probe.py是有限串行任务。后续32/64-update队列等待它退出后才使用GPU0；继续原来的训练协议，不会偷偷将新precision profile用于续训。
