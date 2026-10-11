# EXP-017/v1 — 变长真实动作的固定位置布局候选（CPU）

2026-10-11 09:00 HKT，Judge。**当前EXP-017已完成并验收CPU候选；夜间监督结束，无待执行GPU或训练任务。** 下文保留已执行任务书，后续方案仅为草案。09:00后项目最多3卡，GPU3/4他人进程不操作。

## Track / Parent / Question / Hypothesis

Branch A/B协议准备；Parent冻结V3 full37 Global布局、EXP-015真实17-span脚本和已复现future text_len风险。问题：能否给真实变长动作定义显式action-content-independent Global布局，同时在冻结等长A/D输入上逐字段复现原V3，并在C1/C2裁剪后隔离未来文本内容和长度？假设可用冻结模板固定语义坐标，而物理row索引随当前实际文本长度变化；只作CPU候选，不预设模型效果无损。

## Scope / Inputs / Controlled Variables

仅四固定train scene的EXP-015 action_script17及冻结V3/EXP-014 full37 canonical A/D metadata。可用已有tokenizer在CPU获得真实句长，但不加载text encoder/VAE/DiT或调用模型forward。旧fixture只作布局/长度参考，不把旧I0/embedding作为新训练输入。所有来源SHA记录。

保持SingleI0、Global/MM-RoPE/PRoPE意义、native时间、strict causal、own-action/current-prefix、clean raw KV、12+5分块不变；候选独立文件、独立说明。禁止改正式runtime文件或已有V3证据，禁止默默增加zero prefix/padding、更改action attention来掩盖位置问题。

## Candidate Design / Execution

允许研究一个明确候选：
- 使用每scene冻结canonical full37布局作为**固定坐标模板**，模板在未来控制输入之前确定，不从实际未来script总长计算。
- 实际action embeddings保持真实长度、独立逐句编码规则；物理row/span/index随实际文本长度变化，当前可见mask依原规则生成。
- head、I0、audio/video与每latent action的语义位置由固定模板指定；先明确每字段映射，再裁剪未来action/video。不可只平移video而遗漏prefix或相机相关位置。
- 作为显式新输入协议候选记录，不称对任意mixed动作与原生全段重建无损等价。如果无法在时限内做出清楚且安全的实现，提交设计和未完成门，不仓促接训练。

CPU验证核心：
1. 冻结canonical等长A/D输入的所有packed字段/可见prompt/mask相关输入逐值复现。真实文本embedding无需GPU，可复用原冻结fixture验证canonical分支；标清复用只用于回归。
2. 四scene真实script的tokenizer长度、17spans和输出映射明确。用于37布局的未知尾部采用固定占位合同，不能读取真实未来控制信息。
3. 固定C1（stop12），任意改变C2及更远未来的句子内容与长度，所有C1可见token/positions/indices/mask相关元数据不变；同理stop17隔离更远未来。至少包含增/减长度和明显不同content，不只比较shape。
4. 相同已提交history的语义坐标在后续可见窗口扩展后不变，不能把旧KV重新解释成不同RoPE坐标；说明camera/PRoPE处理和raw KV复用前提。
5. action routing与future物理删除检查、越界/重复indices检查。新增mask协议若无法复用既有逻辑，必须标未通过并停止，不自己改attention。

## Budget / Stop

4 CPU affinity、≤10分钟墙钟、≤100MiB新增文件、磁盘≥60GiB；08:57停止新实现/测试并交付实际证据，09:00交接。所有GPU/模型forward/encode/decode/训练为0。不下载、不扫方案、不加训练，不为CPU PASS启动GPU。

## Deliverables / Acceptance / Next GPU Proposal

独立候选实现/协议说明、CPU命令/结果/源SHA、canonical回归与未来隔离/缓存坐标检查、实际真实tokenizer长度、差异与限制；没有达到的门必须NOT_TESTED/PARTIAL。更新report.md等待Judge。CPU通过仅说明布局工程，不说明mixed-action生成质量。

提出后续有限GPU方案但不执行：先相同冻结canonical状态比较新旧forward（记录dtype/backend），再给真实混合动作条件做有限V3/FM8对照；若需要新的text encode或training，一并单列调用预算。普通FM真实转移适配与AnyFlow finite-map训练分别设计；无D真实GT和四scene局限继续保留。不得恢复已归档AF4/DMD配置。

## EXP-017 CPU最终验收（08:57 HKT）

四scene固定布局候选CPU通过：canonical A/D逐字段复现、未来短/长内容与长度24次独立干预可见布局不变、历史三轴语义位置不变。初版极短future提前拒绝已修复并保留旧attempt证据。0GPU/forward/训练；真实text encoder/DiT/KV数值和视频仍NOT_TESTED，不升级正式模型。下一阶段为未执行草案，canonical双入口若均重算应计8forward而非4，后续14text/156forward/8decode另定预算。[最终审核](submission/experiments/EXP-017_v3_fixed_action_layout/judge/FINAL_REVIEW.md)。

## 09:00交接与下一步

本轮所有获批任务已收口，当前项目0GPU。下一步建议是独立实际模型回归→真实mixed-action文本编码→首scene真实C1下FM30/FM8局部续写，再按结果决定是否扩四scene；完整预算见[Judge下一阶段草案](submission/experiments/EXP-017_v3_fixed_action_layout/judge/NEXT_STAGE_DRAFT.md)。草案尚未变为新执行任务/marker，不启动训练或恢复已归档AF4/DMD。
