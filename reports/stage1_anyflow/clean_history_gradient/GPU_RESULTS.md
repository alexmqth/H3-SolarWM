# 真实33B clean-history梯度探针：通过工程检查，画质待验证

记录时间：2026-10-08T09:02:17.371875+08:00。GPU0独占，reserve6，fresh full-scope rank8；A/chunk2，同一初始化、噪声和AnyFlow四类样本。

| 项目 | detached history | full history |
|---|---:|---:|
| 反传耗时（秒） | 126.791046 | 215.038836 |
| 梯度范数 | 0.001519 | 0.001573 |
| GPU allocated峰值（MiB） | 37009.461426 | 40973.345703 |
| GPU reserved峰值（MiB） | 39796.000000 | 42854.000000 |
| 额外带图历史前向 | 0 | 8 |
| CPU raw KV峰值（MiB） | 5403.4423828125 | 10806.884765625 |

四个sample的raw loss逐项相同；weighted loss均为0.118051555008。full与detached梯度差范数0.000651273，cosine=0.911813，309个梯度tensor不同；QKV/out/FFN/refiner四组梯度均有限且非零。

随后执行一次full-history AdamW更新，312个B投影非零，原visual/time/action参数逐张量不变。含准备与比较共356.36秒，单次更新allocated峰值40973.35MiB，reserved峰值43176.00MiB。没有OOM。

这证明在当前硬件上可保留历史梯度，并证实detach改变了参数导数；它不证明这是画质失败的原因，也不证明视频已改善。CPU KV包含目标用detached cache与prediction用graph cache，未包括checkpoint/offload的其他CPU激活。

探针bank不是可恢复训练checkpoint。后续训练从原始visual/action初始化和零B bank重新开始；不加载本探针更新权重。见[full-history候选](../full_history_candidate/README.md)。本候选仍是现有cache语义的梯度实现，不是官方融合两流算子。

原始数值：[gpu_probe.json](gpu_probe.json)。冻结的preflight与CPU审计文件保持不变。
