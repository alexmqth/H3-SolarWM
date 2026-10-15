# EXP-023/v1 补充02：训练前向与checkpoint反向重算保持相同attention协议

2026-10-15 HKT，Judge修订并授权执行；本补充优先于先前“工程梯度通路已验证”的结论。补充01统一source RGB与预算继续有效。

## 已核实风险

EXP-022 profile_gpu.py的candidate只在forward内开启current_prefix_feedback；函数返回后context恢复ChunkAttention.attend，loss.backward在外部调用。冻结gradient_checkpoint_forward在offload分支使用torch non-reentrant checkpoint，重算block时调用causal_control.attend，动态查找类方法。恢复后的方法缺少common-prefix读取当前video的边。因此“forward完全相等”不足以证明backward计算了同一函数的梯度。旧step02不可作为正确V3梯度协议证据；梯度数值偏差和影响尚未量化，不称旧模型能力已被证明失败。

## 立即执行的修复与门控

1. 编码pilot及后续统一RGB编码与此问题独立，可按补充01继续，不重跑已完成的正确编码。
2. 仅在EXP-023实现明确的训练context生命周期，**覆盖可微forward、loss构建、整个loss.backward及其checkpoint重算**。退出后恢复原函数。调用helper不得再嵌套开启同一全局patch context（旧verify_one_edge_change不支持对已patch方法再次验证）；明确由一个外层owner管理，或提供等价的显式绑定实现。不能修改冻结EXP-022或改变prefix/time/action attention语义来规避。
3. CPU针对真实ChunkAttention/current_prefix逻辑做小张量、非reentrant checkpoint+offload的梯度回归：固定相同状态/权重，正确context下checkpoint on/off的输出与梯度在声明dtype容差内匹配；构造能暴露旧forward-only生命周期错误的负对照，记录调用/重算拓扑差异。选择使prefix反馈影响可训练参数的非退化输入，不以零梯度或未实际重算作为通过。CPU test通过后可直接进入真实训练门控。
4. 在EXP-023**第1次已预算的真实训练forward/backward**中埋点，区分初始forward和backward重算，记录真实发生重算的层、实际attention实现及必要mask语义（common-prefix→current-video边保持）、cache/历史不变。不要求所有50层都重算：冻结前层可能无梯度；必须有真实backward重算记录，并证明所记录的调用没有退回原attention。检查base/released LoRA无梯度、adapter finite/有效梯度。
5. 上述真实门控在optimizer.step前验证。通过后该步计为128updates中的第1步，自动继续剩余127updates和既定评测；**无需再等Judge marker**。失败则不执行optimizer.step，保存forward/backward账本与证据并停止训练，不自行重试。checkpoint/metadata须记录补充02与修复实现SHA。

## 预算和结论范围

不新增GPU额度：门控使用原128backward/updates、440顶层DiT中的首步及既有回归余量；失败调用照计，不重置账本。补充01的70VAE encode/13800GPU秒及其余上限不变。本轮仍从零增量初始化，不读旧step02。

CPU检查、真实重算hook与第一步结果写入GRADIENT_PROTOCOL_CHECK及完整日志。前向等价/重载可复现/梯度finite分别记录，不能相互替代。修复通过仅支持所测协议梯度路径，最终生成效果仍由固定评测判断。
