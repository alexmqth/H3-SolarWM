# Judge阶段一审核：EXP-005 / v1

2026-10-10 HKT。**决策：accept（仅CPU实现/协议准备）；GPU阶段未批准；SW-G/SW-L生成能力NOT_TESTED。** V3 Original Feasibility Baseline继续保持EXP-003的正式验收身份，不用候选覆盖。

## 已完成与独立复核

Worker提交显式分块、窗口入口、Local位置视图、14项CPU测试与协议说明。Judge独立阅读冻结EXP-002/003入口、实际runtime raw cache/current-prefix、原生MM-RoPE与packed构造，核对SolarWM参考和新实现；独立复跑最终套件：**14 passed in 3.74s**，日志[在此](cpu_tests.log)。环境为Python/PyTorch CPU、fp32、`torch.nn.functional.scaled_dot_product_attention`默认CPU调度、最多4线程，未固定单一kernel，详见[environment.json](environment.json)。无33B模型加载，无GPU forward、VAE或训练。

| 检查 | 审核结果与边界 |
| --- | --- |
| 显式partition / 大于6块 | `[0,12)`后每块5；结构性测试到10块，不再依赖旧index 2–5名单 |
| 缓存索引/层/容量 | 真实H3ChunkCache(max_history=5)，精确最近5祖先；拒绝空cache、缺层、错indices/rows/commit计数。小张量实际commit检查多次淘汰与容量稳定 |
| 历史不可变 | 采样前后entry/storage/version保持；Local读取共享raw K/V，不原地改历史；旧RGB通过已有append-only函数检查。未读取或克隆33B历史cache |
| 首淘汰前SW-G回归 | index2–5真实旧/新入口、相同toy模型边界与真实attention路径，CPU输出逐元素一致；实际调用的current/audio/prompt/anchor、packed坐标/actions、native timesteps、prefix时间flag及路由逐项相同 |
| 未来隔离 | 真实parking37条件按可见范围物理删除future action/video；结构性长输入额外拒绝未裁剪未来行；没有用全33B模型做干预生成 |
| Global/Local位置 | 原生非均匀网格正确；首淘汰前Local复用原对象；淘汰后仅video映射，prefix保持；实际H3 RoPE展示video-to-prefix logits可改变；提交canonical Global位置元数据 |
| 首次淘汰 | 旧EXP-003末C6未commit；C6 clean commit淘汰C1，C7看C2–C6。CPU已验证，GPU尚未触发 |
| 拒绝错误协议 | 缺current-prefix上下文、非sigma0提交、错误分块和未经认证post37调用拒绝；真实长输入/GPU入口继续关闭 |

## 审阅修复与有限结论

本轮审查修复了Local canonical RoPE校验跨CPU/GPU存储位置的问题，并增加缺current-prefix上下文的入口门禁。测试进一步核对实际传给模型的参数，避免只凭审计字典或忽略kwargs的假模型声称条件不变。跨设备实际执行仍需未来GPU阶段验证，CPU通过不作CUDA证明。

原生H3网格不是等距latent index；第一次窗口起点b=12，Local重建并非统一减常量。固定Global prefix会改变跨模态相对位置，历史raw K/V又包含此前层/上下文依赖。因此Local不能称无损等价或预设更好。当前冻结H3没有SolarWM camera PRoPE输入，本轮不增加camera模块。

**尚缺经认证的>37latent真实条件fixture与生产GPU runner。** 默认长packed重建会改变原prefix/视频坐标；现有长输入仅结构toy，实际入口禁止post37 GPU调用。需冻结新增action/位置/噪声外推，逐值保留原37条件和噪声，再实现独立授权/逐调用预算账本。任务书允许将此项明确列为阶段二审批前置，故接受本次CPU交付，不宣布运行就绪或模型能力通过。

完整33B首淘汰前输出误差、低精度/backend差异、A/D方向/切换、人物结构和长时KV/系统成本均为NOT_TESTED。五祖先仅约束历史video KV；action prefix、latent/RGB和VAE全前缀解码仍可能增长。

## 下一阶段决策

[GPU_PLAN](../GPU_PLAN.md)仅为提案：G0首次淘汰前回归 → G1 SW-G C7/C8 → 独立L1对照；核心上限281forward、9VAE、1.70GPU小时、1卡、项目≤3卡。C9默认不启用。输入认证和runner门禁未完成前不批准GPU，后续每阶段须独立记录批准范围。

当前CPU任务收口，Worker不继续GPU或训练。普通画质缺陷按可行性尺度记录；严重持续失败时停止相应候选，不追加参数扫描或为微小收益反复实验。[V3-FM8/V3-AF](../FUTURE_ANYFLOW.md)仅独立设计，旧AnyFlow产物不作V3-AF结果。

## 报告归档说明

Worker原始报告/README/PROTOCOL/metrics逐字节保存在`worker_*`快照。原报告把历史“晚上8卡、9点后3卡”按当前日期推到10月11日；Judge现行任务不据此新增或重置资源授权。当前CPU额度0GPU，后续提案单卡、项目≤3卡；只有明确的新任务授权可改变。本次实际GPU使用为0，日期表述未影响任何作业。
