# Full-history TF-AnyFlow39：条件启动的受控16-update候选

目标仍是39f A/D少步动作与视觉效果。当前detached/full-scope AnyFlow16及同容量FM16全部评测完成，两者均未过动作gate。真实历史梯度探针已通过，fresh16训练已完成，step16视频正在评测。

本队列等待真实33B历史梯度探针**成功退出**后，才开始fresh 16-update训练，并评测A/D4/8 steps/chunk。如果探针出现OOM、loss不一致、梯度异常或冻结约束失败，本队列停止，不自动更换配置重试。

唯一训练数学变化是--history-gradient-mode full：梯度prediction的clean-history K/V有计算图，目标三次前向仍no_grad。使用独立不可变读cache，临时capture前向后seal/clear。其余初始化、数据、time冻结、全50+2 QKVO/FFN rank=alpha8、FP32、LR3e-5、logical batch4、seed/noise、RGB dual、action routing/feedback、chunk5/history5、shift2.22均固定。**不加载探针更新后的bank**；探针仅验证梯度和显存。

脚本将独立核对真实step00四套adapter、CPU/CUDA/logical RNG、初始固定noise验证samples，以及16次action/chunk/sigma序列与detached实验一致。每次历史额外前向次数必须等于chunk_index×logical_batch；第0块无需历史。CPU KV需记录detached目标cache与graph cache之和，额外checkpoint CPU activation不算KV。

本实现保留当前cache推理语义，不是官方融合两流算子。源码是已通过小H3数值差分、前向逐元素、10项相关测试和sealed连续/恢复检查的独立副本。真实GPU探针已通过，正式16次训练已完成、step16视频待完成，16次仍是pilot，不宣称补齐梯度必然改善画质。

评测要求完整39f/24fps/H264、same conditioning逐张量、真实bank/AnyFlow元数据和12/24 noisy+3 clean commits；视觉与A>0、D<0、A-D>1同时验收。当前无新124f计划，Stage2暂缓。

## 独立GPU轨迹的解释限制

实际step00四套adapter、CPU/CUDA/logical RNG、初始validation samples及314项冻结输入检查均通过（gpu_initial_audit_early.json）。但独立GPU运行的chunk0梯度不逐bit相同：step1 loss一致，grad norm为0.000945614与0.000944774；step2已出现raw loss差异。此时full分支没有历史可反传，所以不能将这类微差归因于历史梯度。GPU1同权重/同sample/no optimizer重复诊断已确认两次detached反传也有数值差（cosine0.99991），full-no-history处于相近量级；具体算子未定位，见[结果](../../04_numerical_checks/gradient_repeatability/RESULTS.md)。后续对照是配置/初始化/样本匹配，不宣称GPU优化轨迹在历史首次出现前逐bit相同。

真实16次训练结果与公共验证残差见[GPU_RESULTS.md](GPU_RESULTS.md)。
