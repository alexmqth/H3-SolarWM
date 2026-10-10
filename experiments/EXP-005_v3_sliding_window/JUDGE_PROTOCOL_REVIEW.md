# Judge：三个候选、参考核实与风险

2026-10-10。阶段一CPU审阅；不是GPU能力验收。正式Baseline身份沿用EXP-003发布commit `d039352941708d4c7c757d696009b67c3e0e2468`，用户此次命名为 **V3 Original Feasibility Baseline**。当前目录[mainline/v3/v3_baseline](../../mainline/v3/v3_baseline/README.md)，用户引用的`mainline/V3_efficient_causal/`保留兼容入口。

## 1. 结构与协议对照

| 项 | Baseline | SW-G | SW-L |
| --- | --- | --- | --- |
| 权重/训练 | Original H3 + released Action LoRA，无新训练 | 相同 | 相同 |
| 首窗/后续块 | 12 + 5×5，124RGB | 同一分块规则，显式扩至第7/8块 | 相同 |
| cache | CPU persistent pre-RoPE raw video K/V + stored RoPE | 原结构，`max_history=5` | 同结构保存canonical Global RoPE元数据；读取用临时Local RoPE |
| 第7块祖先 | 未验收 | C2–C6，精确indices[1,2,3,4,5] | 相同祖先与窗口 |
| 当前video Q/K与历史video K位置 | Global | Global，逐值保留 | native Sliding Local时间网格，空间轴不变 |
| prefix/动作/时间 | 原Single I0、own-action反馈、current非action反馈、native timestep | 不变 | 不变，prefix保持Global |
| camera PRoPE | 当前冻结DiffSynth H3路径没有camera输入/PRoPE分支 | 不新增 | 不新增；不是SolarWM camera协议的完整移植 |
| 状态 | 124帧有限可行性accepted | CPU准备，GPU待批 | 独立CPU位置候选，GPU待批 |

0-based chunk k的区间：k=0为[0,12)，k>=1为[12+5(k-1),12+5k)。祖先必须精确等于`range(max(0,k-5),k)`，所有50层一致，不能只检查数量或对空dict做all()而伪通过。

C6采样时cache=[C1..C5]，结束时旧实验不做无用commit；要生成C7须按冻结sigma0协议提交C6，此时才真实淘汰C1。候选不要预先裁减raw KV冒充完成了合法clean commit。

## 2. 已核实的SolarWM与本项目差异

参考SolarWM checkout `ce1da4e7705391eda8eeda6016c0fd3f614b975e`：

- `src/solarwm/backends/minimax_h3/sgf_attention.py`的`H3RawKVCache.history/commit`保存最近5个具体indices；`prepare_sgf_rotaries`为5/10/15/20/25/30latent预构建Local MM-RoPE；`sgf_attention`在读取raw KV后旋转Q/K，再应用camera PRoPE。
- `sgf_attention.py`约159–201行使用`first=max(0,index-5)`、当前5latent与完整W6窗口；prefix旋转保持独立，camera从全局轨迹对应帧选取。`sgf_rollout.py`经专用student、独立prefix layout、4次生成调用，commit用其native时间1.0。
- 本项目`interval_cached.py`为30-step native FM、video `sigma*1000`、audio1000、`fixed_prefix_timesteps=False`、commit sigma0；首块12latent而非5；current-prefix读取当前video；冻结`minimax_h3_dit.py`使用三轴MM-RoPE，未提供上述camera PRoPE接口。

因此只借鉴缓存窗口和按窗口构造位置的方法。不能复制SolarWM的4步schedule、commit时间、prefix分块、camera或把其专门训练student的结果外推到V3原权重。

## 3. SW-L明确的位置定义

令原冻结video起始时间为O，native网格为：

`tau(0)=0; tau(n)=sum_{i=0}^{n-1} (5/3)*[1,4,4,4,4][i mod 5]`。

当前chunk k的最早可见祖先起始latent为b。SW-G使用`(O+tau(j), h_j, w_j)`；SW-L使用`(O+tau(j-b), h_j, w_j)`。b=0时二者严格走同一位置路径；第一次淘汰后b=12，之后b=17、22等。b不是chunk数×固定单位时间。

- prefix里的非action文本、action文本、I0、audio坐标保持原Global位置，native timestep与所有mask保持不变；仅保留的历史video K及当前video Q/K使用Local位置。
- 历史raw K/V不重新计算、不原地旋转、不回写。读取临时生成每个保留chunk的RoPE表，raw K/V引用原张量；commit写入当前clean forward得到的raw K/V，同时保存其canonical Global位置元数据供后续校验。canonical位置标签不代表其隐藏状态由Global上下文生成，必须另记分支/mode谱系。
- camera处理：当前V3无camera输入/PRoPE，G/L都保持无此新增模块。若后续切换带camera的backend，需要另立协议任务；不能把未知camera变换填成identity后称本次单变量对照。

### 为什么不等价整段Local重算

第l层历史raw K/V已经依赖生成该历史时前l-1层的注意力、prefix位置及当时可见上下文。仅重映射本层旋转不会更新这些隐藏状态。即使raw key在第一次RoPE前缓存，也不能推出跨层重新定位可交换。C7还共享Baseline历史，C8开始各分支的历史表示会继续分化。

### 为什么不能预设Global/Local不同或Local更优

对理想纯旋转，若所有相关Q/K同一轴统一平移Δ，video-video内积可保持相对位置不变。但本候选只改变video，prefix保持原位，video↔prefix/action的相对位置会变。另外native网格非均匀，b=12并非5周期的整数倍：重新从局部tau(0)起算不等同统一减常数。需用实际H3 MM-RoPE与相同raw Q/K做CPU差异/不变性检查，再以固定history GPU模型和视频评价，而非按方法名称预判收益。

## 4. 主要风险与判定办法

| 风险 | 影响 | 本轮/审批前处理 |
| --- | --- | --- |
| 默认长packed重建改变text_len与action mirrored origin | 首淘汰前已换了条件，不能归因SW | 前37video/原text/I0/audio坐标与prompt/noise逐值匹配；不通过则不批GPU |
| 新action行的模板和全局坐标外推 | 37之后没有冻结参考，可能改变native语义 | 明示外推规则与来源；A/D、G/L共享fixture；不称默认builder等价 |
| 缺层/错index/重复commit | 缓存账本表面正确但历史错误 | 精确50层、indices、tokens、调用次数、sigma0检查 |
| Local metadata跨CPU/GPU或dtype不一致 | CPU测试通过仍可能无法运行 | 显式统一校验元数据device/dtype，rawKV保持存储；GPU首阶段单独限额 |
| 忘记启用current-prefix反馈 | 候选暗改baseline attention | 入口fail-closed或显式验证所用router；复用冻结路由，不重写mask |
| Local raw KV与全历史重算不等价 | 质量变差并非简单位置精度问题 | 独立候选，报告history来源；不改prefix/time救结果 |
| KV有界被写成全系统有界 | prefix/latent/RGB/VAE继续增长 | 分列CPU KV、进程RSS、prefix tokens及decoder成本；结论仅限历史video KV |
| 30步变8步夹入SW验证 | 无法判断窗口/位置收益 | 本任务锁30步，FM8另立任务 |
| CPU toy attention被当模型回归 | 对全网络/低精度输出过度结论 | CPU只支持局部协议/attention结论；GPU G0再测全模型与C6重放 |

## 5. 当前决策范围

仅实现并审核阶段一CPU。候选具备代码和测试不等于生成能力PASS；[GPU_PLAN](GPU_PLAN.md)按G0→G1→L1分段批准，第9块默认关闭。后续[V3-FM8 / V3-AF](FUTURE_ANYFLOW.md)独立设计，EXP-005不启动训练。普通画质缺陷按可行性标准记录；持续失败方向有限预算后停止，不为细小收益反复实验。
