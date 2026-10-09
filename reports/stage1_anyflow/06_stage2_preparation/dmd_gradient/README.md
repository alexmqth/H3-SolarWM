# D准备：正确velocity符号的DMD与完整H3 FMBS梯度接通

2026-10-09 00:00。新增主代码`causal/dmd.py`，不是对旧Stage2-lite结果的重新命名。当前只有数学和tiny-H3 CPU验证，未越过B/C画质/action前置开展33B Stage2训练。

H3采用noise-minus-clean velocity：`x0_hat = z_sigma - sigma*v`，因此DMD的`fake_x0-real_x0 = sigma*(v_real-v_fake)`。直接套用`v_fake-v_real`会反转该参数化下的更新方向。`distribution_matching_loss`接收保留完整生成图的student endpoint，score和normalizer均detach；返回的endpoint梯度包含mean reduction的`1/active_elements`尺度。

`score_sample`只对detached student输出加噪；`critic_flow_loss`用`noise-student_endpoint.detach()`拟合fake-distribution velocity。函数不决定attention/history/conditions或角色，也不假设任何critic已经收敛。调用者必须让Original teacher与独立critic读取同一noisy sample及匹配条件，不得跨角色复用KV。

[7项测试收据](verification.json)与[日志](tests.log)：

- 与SolarWM `training/sgf.py`在两批样本、不同sigma、masked/unmasked的DMD方向及surrogate逐元素一致；masked NaN不污染有效区域，active NaN明确失败。
- 可解析Gaussian real/fake后验验证梯度下降让student mean靠近teacher；反号负对照远离。没有用自己实现重复自己作为唯一方向验证。
- Critic目标、重加噪sample、teacher/fake score不向student反传；没有用detach后的endpoint冒充完整生成Jacobian。
- 实际tiny-H3 FP32/BF16：先生成两块history并clean commit到CPU KV，再保留current chunk的三段FMBS图；中间执行独立critic AdamW更新，随后以Original双向teacher与更新后critic构造DMD。Student全参数梯度与直接J^T g对照一致，base/critic无附带梯度，历史KV不变。

数学参考：[SolarWM训练函数](https://github.com/Junchao-cs/SolarWM/blob/ce1da4e7705391eda8eeda6016c0fd3f614b975e/src/solarwm/training/sgf.py)；本机实际源码hash见收据。实现、测试同时进入submission主代码，依赖已有`fmbs.py`、`shared_h3_roles.py`及对应tiny-H3测试fixture。

下一步仍需可信B/C checkpoint、真实模型成本与多轮fake-score拟合，再进行完整on-policy效果评测。该CPU通过不等于critic已学到student分布，也不等于D阶段已完成。
