# 后续V3 DMD的CPU协议核查

2026-10-11，Judge准备工作。**CPU_PASS；没有DMD GPU任务或生成能力结论。** AF2训练按原预算继续，不受本检查影响。

执行：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-007_v3_anyflow/dmd_cpu/check_protocol.py`

使用实际H3代码的随机两层tiny模型，teacher/fake/student为三个独立实例，沿用V3 interval_student、12+5、current-prefix、native time、Global位置和各角色自己的detached raw KV。耗时2.613秒，0 GPU forwards。

已检验：

- student实际执行8个有连续计算图的finite maps，每个map velocity和中间状态均收到generator梯度；没有用另一条endpoint replay代替生成链。
- fake-score没有target-time模块，对student detached endpoint做一次独立普通FM更新；该反向不影响teacher/student。
- fake更新后重建自身C1 KV并检测到变化。teacher/fake/student KV互不共享storage。
- teacher与fake在同一detach+renoise endpoint、相同sigma/条件上评分；`fake_x0-real_x0 = sigma*(v_real-v_fake)`符号与归一化一致。
- generator反向只更新student路径；target/QKV梯度均非零，teacher/fake无新梯度，基础权重版本与student历史KV保持不变。
- detached endpoint会被DMD helper拒绝，不能冒称真实生成链梯度。

[执行代码](check_protocol.py)和[机器结果/调用记录/源SHA](result.json)完整保留。随机tiny模型的direction只有约8e-6；这个数值不衡量实际33B可行性、训练收益或质量。使用冻结C1只支持当前块on-policy，不是完整多块自生成历史训练。

## 真实模型阶段的实现风险

AF2的单map checkpoint backward可行不代表8map链显存已验证。teacher/fake/student若分别加载33B并常驻同卡可能超显存；不允许为省显存悄悄detach生成链或在checkpoint backward前替换共享模型的角色权重。优先给角色独立模型/设备，或证明显式卸载保持参数身份与计算图，先按独立任务预算测一次cycle成本。

DMD GPU任务仍须先有AF3视频结果、独立编号和有限cycle预算。旧stage2-lite结果不能作本轮证据。
