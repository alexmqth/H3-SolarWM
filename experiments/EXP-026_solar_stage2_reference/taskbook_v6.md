# EXP-026 plan v6：单样本官方 Stage 2 参考准备

2026-10-16 HKT。用户批准继续，同时要求 Judge 讨论必要性。原 Exp Worker 立即执行 A；B 的具体 GPU 配置/预算由 Judge 审核实际输入后下发，不再向用户重复索要开始许可。plan v6 是任务修订号。

## 研究决策与 ROI

保留一次短片官方参考有价值，但不是 V3 继续研究的前置门槛。研究问题：充分训练的 released SolarWM-H3 Stage 2-fix 能否在原生相机条件下完成短时 causal/KV 生成，其边界连续性、运动及成本如何？只观察一个样本，不据此概括整体性能。它不能隔离 SGF/AnyFlow/DMD 的各自贡献，不能证明我们的语言动作 V3 仅需更多训练，也不是公平的键盘动作模型对照。

不训练 Stage 2，不扩大30–50场景，不重新构建环境或下载模型。若权限仍阻塞，停止此支线等待真实访问变化，主线 V3 不因此等待；不为解锁参考反复换协议/缩分辨率/造数据。

## A：立即执行 CPU/输入准备

1. 先读 v5 Judge 审核与现有 INPUT_MANIFEST，复用已成功的隔离 overlay/工具链/四卡加载补丁。新证据全部写 reference_v6，不覆盖旧尝试。
2. 检查已有数据目录/合法本地缓存，以及标准 Hugging Face / ModelScope 凭据是否已配置，记录是否存在与访问成功，不打印 token、请求头或完整凭据文件。用户未提供新路径时不扫描其他用户私有目录。
3. 允许每个官方提供方一次有超时的实际小文件访问，确认现状；401/403 无新凭据则立即停止该提供方，不重试等价URL。查明官方自包含 example 是否真的有匹配 H3 内容/支持文件；只检查公开资料或已获授权访问，不绕过权限，不以 schema toy sample 冒充正式测试样本。standalone全长SP8路径不自动切换。
4. 访问成功时，按现有seed42官方选择顺序固定一条测试记录，下载最少index/support/shard，记录revision/URL/字节/SHA；先核实shard大小，未知或超过预算则停止下载提案。CPU实际reader验证I0/text/latent/camera/intrinsics/silence/encoder contract，保留真实来源。
5. 配置基于官方158帧、1344×768、24fps、47encoded/50rollout/decode47、4NFE、W6、Sliding Local、原生camera/time/exit-cache协议和v5 SP4/DP1/shard4；仅一条EMA结果，不能意外跑live+EMA或额外样本。从实际入口核对denoiser、commit、codec及可选文本/图像编码调用数。普通模块导入和CPU校验不得初始化CUDA。
6. 写 INPUT_STATUS.json、WORKER_REPORT.md 与 GPU_PROPOSAL.md。若缺数据，列出最少缺失路径及官方访问入口，无需空转满预算。若输入就绪，立即消息Judge提交可执行命令、代码差异、模型/codec并行与显存风险、计数/预算/看守，等待Judge的B授权。

A预算：CPU工作≤30分钟，≤8 workers、RSS≤96GiB；下载≤20GiB、新磁盘≤30GiB、保持≥200GiB空闲盘；单网络请求超时≤60秒、无进展5分钟停止；0GPU/forward/codec/training。任务只做一次输入解除阻塞与单样本准备，不反复验证已通过的加载。

## B：单样本推理方案上限（尚未授权GPU启动）

建议4张空闲L40，最多一次40分钟=9600GPU秒；最终预算需以Worker的CPU就绪证据签发。若预计超出则先报告。只有一条158帧EMA输出，无额外训练/种子/消融。所有加载、首次kernel编译、采样、commit、decode计入预算。限定超时/显存保护，OOM或协议错误即停止，不自动重试；夜间总≤8卡，09:00前释放四卡，日间≤3卡。

交付视频及可复现配置、日志/逐阶段耗时、采样NFE与额外commit计数、GPU allocated/reserved/物理峰、CPU KV/RSS、实际淘汰/边界与运动观察。仅有限forward不能判生成能力通过。无需在这一步证明论文级指标或所有相机动作正确。
