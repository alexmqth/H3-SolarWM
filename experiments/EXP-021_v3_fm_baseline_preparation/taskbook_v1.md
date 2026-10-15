# EXP-021/v1 — 普通FM因果适配与固定评测集准备（CPU阶段）

2026-10-15 HKT，Judge授权Exp Worker执行。Research Track：Branch B；Parent：冻结V3 + EXP-019固定动作布局。当前授权仅CPU准备，GPU forward/encode/decode/训练预算全部为0。

## 本轮问题与假设

能否用现有真实动作数据、episode隔离评测与保持V3条件协议的普通FM训练入口，形成可执行且成本可控的小数据拟合实验？假设因果适配值得验证，尚不预设训练一定能解决工业背景跳变或动作切换。

## 输入、控制变量与允许改动

读取EXP-013至020的数据/训练入口/冻结证据、真实ABot数据与官方H3/SolarWM本地源码。保留Original H3 + released Action LoRA、Single I0、native time、Global RoPE、own-action/current-prefix反馈、clean raw KV、12+5分块和已验收布局。训练目标为普通FM速度场，不含AnyFlow target-time finite map或DMD。

本阶段允许新增数据manifest、CPU校验与独立训练入口草案；不改冻结代码/权重，不执行模型调用，不下载大数据。若既有训练入口不兼容，列出最小修改，不为凑预算改变目标分辨率/时序。

## Worker交付与顺序

1. 盘点实际可用真实56帧以上clip和动作时序，明确独立episode数量、Constant/Switch覆盖、缺失/排除原因；选32条可靠训练clip作为首轮拟合候选，上限128条。不足32就如实给出数量/补数据成本，不重复clip冒充样本。
2. 固定6–8个评测场景的候选清单，按episode与训练隔离，固定seed和动作脚本。已有四train+两个已观察validation只能标为训练/开发回归，不能冒充盲测；新增候选只凭数据质量选择，不按模型生成效果筛选。数据不够先交最小可用集，后续30–50场景不自动启动。
3. CPU审查FM训练入口：loss/velocity目标与sigma、GT clean commit与梯度边界、当前块未来隔离、released LoRA与新adapter叠加方式、可训练层/参数量、优化器状态及checkpoint再加载。不得用旧AF32updates初始化来声称纯FM基线。提交最小接口/配置草案与有意义的CPU测试；无法CPU验证的反向和数值事项明确待测。
4. 提交下一GPU阶段的独立建议：先1–2个完整forward/backward/update及保存重载profile，再依据实测吞吐决定小数据训练更新数；定义checkpoint0/训练后匹配NFE、GT与self-generated history、constant/switching验证。把编码、训练、验证、失败调用全部计入预算，列清显存/CPU RAM/磁盘和硬停条件；不能凭当前推理吞吐假称训练成本已知。
5. 附一页SolarWM官方Stage2-fix标准相机推理准备情况（已有权重、官方输入、命令、资源缺口与有限GPU预算草案），单列任务。CPU核对EXP-020的56帧/17动作、精度与官方推荐设置差异；不针对天空伪影铺开消融。

## 预算、验收与停止

CPU≤4线程、RAM≤16GiB、新增磁盘≤2GiB、首轮≤30分钟；0GPU/0训练/0模型调用，不执行完整33B CPU推理。找到足以设计首轮profile的材料即可提交，不要求穷尽全数据。来源/时间对齐不明、episode不足、协议不兼容时报告缺口，继续完成独立可做项；禁止自行扩大预算。

交付README、DATA_MANIFEST/CPU_CHECKS、TRAINING_PLAN、GPU_BUDGET_DRAFT、WORKER_REPORT及根report.md；实物留本EXP目录。成功标准是可审核的样本来源、协议、候选入口与预算，不是模型能力PASS。Judge审核后另签GPU授权，旧EXP-019/020 markers不能复用。AnyFlow/SGF+/Memory与新GPU实验不在本阶段范围。
