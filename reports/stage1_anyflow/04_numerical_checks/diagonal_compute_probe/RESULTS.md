# Diagonal AnyFlow：真实33B只读计算检查完成

归档时间：2026-10-08T12:21:07.861066+08:00。本次没有optimizer update，未修改主trainer或运行中的冻结runtime，不是画质实验。

同一个已训练shift12 step32、A/chunk2、完整历史梯度、logical batch4，先重复两次原始反传，再运行一次仅省略r=t零系数有限差分的候选。全部sample sigma/r/raw loss/weight/adaptive scale/weighted loss相同；CPU/CUDA RNG均未变化；冻结visual/time/action及可训练bank参数均未变化。

| Pass | wall s | GPU allocated peak MiB | current forwards | gradient norm |
|---|---:|---:|---:|---:|
| reference_1 | 239.379 | 40974.042 | 16 | 0.022554744 |
| reference_2 | 236.922 | 40974.060 | 16 | 0.022547039 |
| shortcut | 206.748 | 40974.395 | 10 | 0.022562345 |

| 与reference_1比较 | gradient difference norm | cosine | max absolute difference |
|---|---:|---:|---:|
| reference_2 | 0.00015800933 | 0.999975511 | 2.9544226e-06 |
| shortcut | 0.000163625082 | 0.999973751 | 2.71054159e-06 |

候选本批比两次reference平均耗时减少13.19%。current forwards下降37.5%，但完整历史重建和backward未减少，不能把前向计数比例当训练加速比。三次按固定顺序、与GPU0/1共享CPU/offload主机，未完成随机交错多次benchmark，时间只是一次观察；显存基本不变。

候选梯度差异与重复reference处于相同量级，但不是CUDA逐bit等价证明；单批结果不证明长训练轨迹相同。CPU20个真实小H3 FP32/BF16、detached/full-history cases已验证loss/weight/all gradients逐bit相同且cache/RNG不变。跳过的finite_difference_norm记录null，不能当0。

当前仅保留独立候选，主trainer继续原四次前向协议。完整证据[gpu_probe.json](gpu_probe.json)、[cpu_equivalence.json](cpu_equivalence.json)，脚本与331项输入hash一并归档。Stage1画质/action gate仍未通过。
