# FMBS 接入实际 H3 前向：CPU 梯度验证

**已把三段 flow map 接到主代码的 H3 chunk_forward，但没有启动33B Stage2训练。** 这不是可信causal checkpoint或画质门槛的替代。

维护入口为`H3-World/code/causal/fmbs.py`；本目录保存同一版本的源码及测试。`simulate_h3_chunk`固定当前chunk的conditioning和detached历史KV，使用真实sigma网格中的相邻端点，执行`1→t→r→0`，跳过零长度映射，FP32累积state且不detach当前三段中的任何一段。该模块没有teacher/critic/DMD更新器。

与之前只验证通用数学primitive相比，本次调用实际H3 DiT、action rows、AnyFlow时间条件、LoRA、CPU KV及checkpoint/offload代码。使用随机小模型、12个latent、5+5+2 chunks：前两块通过完整8-step推理自生成，再clean commit；最后两帧执行三段可微FMBS。历史KV固定，参数导数只针对当前chunk的映射，不反传穿过生成历史。

以下五类、六项检查通过（含FP32/BF16 backbone；state始终FP32）：

1. 普通计算与checkpoint/offload输出完全一致，参数和起始noise梯度一致；历史KV值/version/commit保持不变。
2. autograd参数方向导数`0.00521061`，固定相同历史KV的中心有限差分`0.00515580`，相对差约1.05%。
3. 首段detach负对照输出不变，但参数梯度差范数为`0.00124804`，约为完整梯度范数的24%。因此单次/部分replay不能冒充完整三段Jacobian。
4. 完成当前forward后，调用者提交下一块并淘汰最早缓存，checkpoint backward仍得到相同梯度。实现保存独立entry列表快照，避免反算误读推进后的窗口；不复制大KV tensor。
5. 无AnyFlow条件、混入commit/time参数、非detached历史及重复sigma网格会拒绝；边界区间只执行实际存在的一或两段。

通用`shortcut_intervals`与`simulate_chunk`去除docstring后的AST和此前经45组官方输出/梯度对照的primitive逐项一致，因此本次没有重复那45项数学测试。新增的是H3集成与缓存生命周期测试。

首次运行四项测试在history setup停止：手写shift公式把起点算成`1.0000000000000002`。已将测试fixture的协议端点明确固定为1和0，保留H3本身的范围检查。修正后`6 passed in 5.25s`；失败未涉及预训练模型/GPU。

代码在[这里](fmbs.py)，测试在[这里](test_fmbs_h3.py)，数值/源码hash/测试命令见[verification.json](verification.json)。小模型使用固定合成anchor，不验证新的RGB VAE流程。

后续D仍需冻结双向Original teacher、独立fake-score critic、同一生成样本的score差及DMD surrogate，并保证共享backbone在backward前恢复student的参数/adapter角色。快照只隔离KV列表，不能允许修改底层tensor或条件。当前也未验证33B三段梯度的峰值显存和吞吐，更未证明动作/画质改善。B/C前置条件继续有效。

完整最终测试输出见[pytest.log](pytest.log)，执行状态与最终源码hash见[pytest_execution.json](pytest_execution.json)。BF16分支完整梯度范数0.00521059，detach首段梯度差0.00124696，checkpoint/offload一致性同样通过。
