# 隔离候选：每个chunk的公共prefix读取当前video

2026-10-08 22:50。状态：CPU preflight与真实33B的18状态探针全部完成；后续12点delta cosine0.028777→0.008548，候选未恢复动作几何。见[结论](INTERPRETATION.md)。没有训练或候选生成视频。

改动包括两个已在首块2×2消融中分开的因素：video直接action绑定从past/own改为own，以及common non-action prefix query允许读当前chunk video。保留current action→own-frame video反馈、video→past raw KV。公共prefix不读历史video KV；每次chunk forward独立计算prefix，已提交历史不被当前/未来condition改写。没有新增参数、solver/anchor/LoRA改动。

`current_prefix.py`通过上下文管理器隔离启用；AST检查要求复制的attention除一条mask赋值外与原版相同，退出恢复原方法。只允许own action模式、feedback=True和实际action rows；生产默认没有改变。

CPU实际tiny-H3代码路径10项检查通过（FP32/BF16），见[cpu_preflight.log](cpu_preflight.log)：

- 首块输出与统一SDPA的Original directed predicate最大误差0。这是身份正控，不是后续chunk/action质量证据。
- 捕获实际Q/K/V attention mask，核对声明的边；多层可达性中当前输出没有未来chunk action路径。
- 扰动future action embeddings，此前完整输出、clean commit KV逐bit不变。
- 当前action扰动确实改变输出（FP32 RMS0.0040160），历史value扰动也影响当前输出（RMS0.0078549）；不是绕开KV。
- A/D预测只读cache；逐chunk保留原conditions重建全部因果祖先，经历window eviction和末块2帧后，persistent/replay完全一致。
- 异常退出后恢复原attention方法。

测试使用随机tiny模型/随机latent与小型anchor，不能代替真实33B RGB anchor协议、GPU内核或视频验收。完整祖先重放是逐块用原条件重建，**不是**把全部历史共享一个可被当前video改写的prefix；后者与persistent cache不等价。未做训练/backward检查，本候选当前仅用于no_grad诊断。

真实33B探针在原零更新H3+released action LoRA上，固定step00-generated endpoints/history/noise，比较现行causal、隔离候选、Original。两个自然scene×3chunks×3sigmas，每组只换当前A/D；两种causal路由各自重建KV，prefix/anchor/teacher output hashes追加记录。首次首块应为Original身份正控，后续chunk才是新证据。没有从旧FM48训练继承候选，因此不把结果当成训练提升。

源码依赖源项目B的冻结runtime、公开数据编码缓存和33B权重；提交包只存源码/收据，不包含这些大文件。完整同状态baseline资料见[FM48几何结果](../../01_real_video/real_abot_fm/FM48_GEOMETRY_RESULTS.md)。
