# EXP-025/v1：全层普通 FM、C1/C2 完整覆盖的小数据拟合

2026-10-15，Judge授权原Exp Worker立即实现并执行本任务。Research Track：Branch B。Parent：冻结V3、EXP-024修复后full50 rank8 C1/C2训练入口。正常CPU自检与首个真实训练门控通过后连续执行，不等待额外marker。

## 唯一研究问题

比起EXP-023的tail8/C2-only最小配置，全层QKV rank8并训练C1与C2，在有限真实数据预算内能否带来可见的视频/动作收益？容量与覆盖同时改变，明确为联合基线方案；不能单独归因容量不足。EXP-024仅证明工程可行，不证明训练质量。

## 冻结协议与输入

Original H3+released Action LoRA冻结，新零增量full50 QKV rank8 adapter，10,752,000参数。不接续023/024已更新adapter。复用023的32train+2dev、统一source RGB的编码、SELECTION和固定四评测clip，验证实物SHA。dev为已观察episode，非盲测。Single I0/native time/Global RoPE/own-action/action feedback/current-prefix feedback/strict chunk causal/clean persistent raw KV/12+5/SDPA/h3_fp32不变。无新数据编码/下载、无新LoRA架构、AnyFlow、DMD或Memory。

## 训练

使用024 coverage.py的真实C1/C2路径。32clip×每clip C1/C2各一次×4epochs = **256 optimizer updates**，C1/C2各128次，每clip每block各4次。seed13 balanced_block_order、每次新独立噪声和sigma~Uniform(0,1)，排除端点；batch1，无梯度累积。AdamW lr1e-4、betas(.9,.95)、weight_decay .01。每block loss按元素mean，C1/C2等频；C2每次按当前adapter无梯度clean commit GT C1，历史不反传，不跨update复用KV。没有self-history训练，因此不能声称解决exposure bias。

新增实现全部放本目录，不修改已冻结023/024源码。记录命令、代码/输入hash、实际模块形状、训练顺序、逐update sigma/loss/梯度/资源。每次更新沿用optimizer前finite梯度/冻结参数/路由门控，C2完整KV摘要检查；记录首个C1/C2完整50层重算门控证据，后续摘要。保存0/64/128/256updates checkpoint与optimizer/CPU+CUDA/数据顺序/噪声RNG；阶段消息回报，不在中间追加视频。checkpoint保存/CPU重载核对须纳入实现。失败不自动重复已消费GPU采样。

## 执行顺序与评测

1. CPU准备：调用生产路径检查全层adapter保存/加载、真实34条件、256顺序配额、采样器加载50层adapter而非旧tail8硬编码。复用已通过媒体原子RGB保存/显式ffmpeg/BF16 raw-byte身份规范。freeze完整来源清单和分阶段预算。
2. 真实零adapter入口回归：同一固定train clip、同sigma/noise，比较新增adapter disabled/enabled的C1 current、GT C1 clean commit、C2 current，共最多6顶层DiT；KV150项/velocity按冻结精度检查，数值不符记录backend并硬停，不偷偷改协议。
3. checkpoint0与checkpoint256各生成固定4clip GT-history（GT39+新17）和2dev完整self-history（全56），共12片、每chunk FM-8NFE；同I0/动作/初噪声、采样schedule，与023选样一致。全层adapter的零初始化前视频也实测，避免无证据复用旧结果。C1/C2按同seed端到端生成；自生成历史前后不同，不能声称同状态C2消融。
4. 固定probe：4clip×sigma .25/.75×C1与C2×前后，共32个velocity probe（每checkpoint16个）。C1空历史；C2共享每clip对应GT clean commit。统计分别列C1/C2与train/dev、均值比和逐项相对变化，不将随机训练loss当学习曲线。
5. 前评测后自动训练256updates，再固定后评测，CPU制作6组三列GT/0/256对照视频与contact sheets，解码/SHA/56帧24fps核验后提交report和消息。

## 硬预算与资源

基于024一对更新约70秒（含摘要等开销、扣加载），128对约8960秒，加评测/回归及缓冲，**全任务最多13800 GPU进程秒、576顶层DiT尝试、256backward/256updates、12VAE decode、0VAE/text encode**。预计严格计数：训练384DiT；前后视频140；C2 probes含每clip commit共24；C1 probes16；零入口回归6，总570；余6仅实现计数余量，不授权新增样本/重试/搜索。若实现需要更多调用，GPU前报告Judge。

阶段：回归≤400秒、pre≤1400秒、train≤11000秒、post≤1400秒；同时受总13800秒上限约束，失败/加载/保存计入。CPU媒体制作0GPU。训练/评测顺序单张启动时空闲L40，优先GPU0，最多1卡；不抢他人进程。日间项目≤3、夜间≤8的上限仍有效。外层watchdog低于各阶段预算并留10秒kill余量；为post保留1400秒及未用其他阶段费用，不可把全部总预算耗在训练。

训练GPU allocated≥44GiB、RSS≥128GiB、来源错误、未来泄漏、非finite、缺层/冻结参数梯度、KV改动、OOM、超时或关键异常均停止并消息Judge；不自动重试、不换rank、不缩分辨率、不延长更新。到预算提前结束保留checkpoint和准确计数，不能写满256。媒体失败仅从已存RGB CPU恢复，不再GPU采样。

## 验收与决策

执行正确/目标拟合/视频质量与动作分别结论。完整看全部生成帧，检查人物结构、ghosting、边界、运动是否减少、动作切换。当前非同状态A/D反事实，不报告不存在的方向正确率。每片报告NFE、forward/commit/decode/采样/端到端耗时、GPU allocated/reserved峰、CPU KV；训练时间单列。学习改善须固定probe与视频支持，MSE小降不等于画质收益。

本轮结束即做ROI决策：若没有清楚视频信号，不自动再追加rank32/更多epochs/AnyFlow/DMD；可以停止此候选或转向有依据的诊断。256updates不预设充分训练或收敛，负结果不等于架构失败。不覆盖正式V3。

## 交付

WORKER_REPORT.md、训练/评测代码与冻结来源、CPU_PREFLIGHT、回归和首步梯度证据、逐update日志、checkpoint元数据（大tensor本地）、累计预算、固定前后指标、12原片/6对照及视频SHA。根report.md更新并消息通知Judge；不操作Git。Judge阶段审核与push。
