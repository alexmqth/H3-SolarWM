# E1：Original局部动作信息流与窗口正控

**2026-10-09 05:09最新：** [完整条件校准与局部结果](CONDITIONING_RESULTS.md)已收尾。整窗口正控恢复，5+5+2仍No-Go。所有本轮GPU作业结束，无训练。下一项为更粗12latent窗口的多历史检验，已登记尚未启动。下方04:16描述为历史实现/时域记录。

2026-10-09 04:16。T1/T2已隔离实现，CPU实际tiny-H3的9项测试通过；真实VAE审计完成。两组Original-prefix/T2停车场局部正控已在GPU0/1启动，尚未完成或验收。没有optimizer update。新几何探针已准备，必须先完成并评审局部正控，才允许启动。

实现见[runtime/code/causal/local_topology.py](runtime/code/causal/local_topology.py)：T1每层重算允许读取历史video KV的prefix，过去video KV保持不变；T2每次调用重算整个可见窗口，不存可变history hidden cache。所有分支在refiner前物理移除未来动作与视频行，保留预先冻结、与未来action内容无关的全局位置。停车场A/D词表每帧10行、布局完全相同；不能把这个位置协议泛化到未来任意长度动作文本而不做适配。

[cpu_tests.log](cpu_tests.log)与[测试源](runtime/tests/test_local_topology.py)覆盖Original实际mask逐项对照、首块identity、future action内容/数量删除、过去KV使用/只读、逐祖先replay与eviction、T2动作/sigma变化及历史latent不变、上下文异常恢复。FP32/BF16均覆盖；CPU测试不是33B动作效果证明。真实33B首块T1/T2已各一次取得max_abs=0，独立repeat=0，后续chunk收益仍待测。

## VAE时域证据

[原始审计](inputs/preparation.json)：两个真实ABot片段，在RGB frame17/34之后分别反转像素，过去5/10个latent的max_abs均为0，共4项。支持本数据协议下GT-history encoder不依赖这些未来RGB；不是所有视频/encoder模式的形式证明。

实际decoder是5 latent主体＋2 latent overlap，RGB overlap=5。停车场A/D参考中，仅扰动latent5及以后，前17RGB帧仍变化：mean_abs约0.00727/0.00705、max_abs约0.518/0.471（RGB范围0–1）。因此整段decode可能回改早期RGB；新视频每个chunk只decode当时已知prefix，追加新RGB区间0:17、17:34、34:39，已展示帧不修改。完整离线decode不可自动当作严格在线输出。

上述审计不证明decoder是此前动作velocity差分失配的根因；velocity探针发生在解码前。它补全了输出层因果性与公平性协议。

## 正控设计

两份history分别为旧Original生成的全A/全D reference，明确不是GT。每次测试固定其中同一份过去，当前chunk fork A/D：同首图、prompt、audio/video noise、全局位置与RGB dual anchor，仅当前动作变化。每个分支30步，12latent分5+5+2；T2只更新当前latent，历史time=clean；known window每个sigma重算。

每份reference输出6条局部片段（17/17/5 RGB frames），保存实际A分支solver索引0/15/27的state供同状态geometry使用。39帧拼接文件明确标记oracle local forks与边界，不是自由rollout。后续geometry会在每一个保存状态上分别计算A/D，而不把A、D两条已经分叉的solver轨迹相减。

Original-prefix就是本轮T2的计算：不能用T2与自身cosine=1判成功。先看局部运动方向/人物结构以及参考是否有效，再决定T1/C0/C1的同状态对照。峰值/耗时是共享主机单次可行性记录，含缓存conditioning的限制；最终速度比较须另做warmup与重复测量。

实时进程以源outputs下receipt的PID + start_ticks及/proc为准，提交包为快照。当前只用GPU0/1，最多3张约束保持。运行中runtime和runner源码冻结，不重启已启动任务。基础权重、reference latents/RGB缓存和模型cache不打包。
