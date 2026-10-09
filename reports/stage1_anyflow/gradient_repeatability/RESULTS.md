# 同状态GPU反传重复性：前向完全相同，梯度有小幅差异

完成时间快照：2026-10-08T09:19:34.951510+08:00。GPU1空闲卡，33B原生FP32 profile、rank8 bank，A/chunk0，没有历史、没有optimizer更新；同模型参数、logical noise/times，顺序执行detached两次与full-no-history一次。

| 对比首次detached | 四类sample记录相同 | 梯度差范数 | 梯度cosine | 最大逐元素绝对差 |
|---|---|---:|---:|---:|
| detached_2 | True | 0.000012636 | 0.999910573 | 2.35546167e-07 |
| full_no_history | True | 0.000013149 | 0.999903209 | 2.50712219e-07 |

三次grad norm：0.000943826、0.000944335、0.000944421；总耗时317.98秒。参数未做更新，原visual/time/action保持不变。

两次完全相同调用已产生梯度差异，且没有历史的full模式处于相近误差量级。因此跨进程初始step0相同并不保证逐bit相同的优化轨迹；BF16反传/归约等实际GPU路径需要按数值容差解释。本探针没有隔离到具体kernel，不把原因直接归于某一个attention算子。

PyTorch2.10.0+cu128；deterministic algorithms=False，matmul TF32=False；flash/memory-efficient/math SDPA均允许。没有修改GPU0正在训练的backend或配置。

此前A/chunk2的full/detached差范数约0.000651、cosine0.912，与这里的chunk0不是同一个梯度点，不能把比值当严格信噪比；历史梯度数学正确性另有CPU原前向数值差分证据。保留GPU0训练；后续必须看明显动作/视觉改善，不用微小loss差宣称有效。
