# D准备：共享H3 backbone的student / Original teacher / critic角色

2026-10-08 23:18。主代码新增`shared_h3_roles.py`；实际tiny-H3 CPU FP32/BF16共6项测试通过。**这不是完整Stage2训练，也不改变B/C的质量门槛。**

此前FMBS已有完整current-chunk Jacobian，但缺少安全角色管理。新实现保留一个冻结H3 base，student与critic各有独立LoRA参数，按enabled flags选择；不把critic/teacher权重覆盖到保留计算图的student参数上。Original released action LoRA保留在base中。

- Student启用既有Stage1 bank及注册的student-only visual/action residual；保留AnyFlow target-time模块。
- Teacher禁用新增student/critic bank及student-only residual，暂时移除AnyFlow conditioner，在no_grad下使用原始时间模块；退出精确恢复对象/flags。
- Critic使用独立bank，student bank/residual/time module禁用。可选tail的QKV/out subset，不要求第二份33B。
- 角色之间不复制backbone参数，不改变optimizer参数引用。冻结base/非所属student参数version有检查，梯度hook拒绝在错误角色下做checkpoint backward。

管理器**不选择attention、action spans、anchor、loss或KV**。真正teacher/critic仍需caller明确采用Original双向attention及一致raw history；不能把causal cached teacher改个名字。各有效模型的KV不能互用。CPU测试以H3原生directed mask、双向全history调用验证teacher，而不是仅检查flag。

## 验证证据

[日志](tests.log) · [收据与源码哈希](verification.json) · [实现](shared_h3_roles.py) · [测试](test_shared_h3_roles.py)

1. 安装非零student bank、非零旧visual QKV residual、非零critic bank，以及刻意扰动的AnyFlow target-time模块后，teacher输出逐元素等于适配前Original reference；FP32/BF16都通过。
2. 两块历史通过student的实际4-step rollout产生，并clean commit到CPU KV；历史detach且在后续读操作中不变。
3. student保留checkpoint/offload的完整FMBS 1→t→r→0图。中间执行teacher前向，再让critic对detached student endpoint加噪，训练一次noise−endpoint FM并AdamW更新。恢复student后，endpoint及全参数梯度与无角色插入的参考一致，冻结base无梯度。
4. student/critic参数无交集，原base对象身份保留；student trainability未被嵌套critic包装误冻结。
5. 错误角色下backward明确拒绝；抛异常后time模块及enable flags恢复；critic可选subset。

初次增加旧visual fixture时误用Stage1 wrapper，被原安装器的重复bank保护拒绝；修正为真实CausalQKVLoRA，并保留失败日志，没有放松生产保护。进一步把visual包装器的base从“role-owned”参数排除，确保底座仍进入frozen audit。

当前只有角色/梯度集成，没有新的DMD梯度实现或真实33B训练；也没有证明fake score拟合student分布或视频改善。下一步仍要在满足B/C局部能力前置后，将Original teacher、独立fake-score objective、正确DMD方向和FMBS生成端接入实际训练。
