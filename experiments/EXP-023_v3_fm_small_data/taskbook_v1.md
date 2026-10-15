# EXP-023/v1 — 32条真实视频的V3普通FM小数据拟合与匹配评测

2026-10-15 HKT，Judge授权原Exp Worker立即执行。Branch B，Parent：冻结V3+EXP-019动作布局+EXP-022可微入口。问题：在协议保持不变的条件下，128次普通FM更新能否改善固定状态速度目标及局部续写，收益能否保留在自生成历史中？不预设成功，不将小数据试验等同充分训练。

## 当前授权与执行分工

CPU准备→编码pilot→其余必要编码→checkpoint0评测→128updates→训练后匹配评测，满足下述客观条件就直接继续，**不用每8/16/32updates等待新marker**。Exp Worker负责实现/运行/报告，Judge审核和Git。准备时冻结来源SHA/配置/选样清单和命令；源码独立存本EXP，不改EXP-022/冻结结果。

## 数据与协议

固定EXP-021全部32train，物化来源EXP-022。只选SELECTION.json中的2dev编码/评测；另6dev不编码。4条已有train编码可复用EXP-016/019，其余28train+2dev共30新窗口编码。两个dev episode曾被研究观察，称开发集，不称盲测。

维持832×480/24fps/56RGB→17latent、12+5、Single I0、native time、Global RoPE、own-action/current-prefix/action feedback、clean raw KV和h3_fp32/SDPA。编码从冻结源索引对应的原始RGB重建，不把H264重编码片的变化悄悄混入既有4条raw-source编码。保存I0/caption/17动作/latent来源SHA；每clip重新构建其正确的固定语义布局和anchor，不能复用别的clip的I0。未来动作截断和历史语义位置沿用已验收协议。

Original H3+released Action LoRA冻结。新增tail8 QKV rank8 adapter，零增量初始化、1,720,320参数；从零增量checkpoint0开始，不从EXP-022 step02或旧AnyFlow继续。lr1e-4、AdamW betas(.9,.95)/weight_decay.01、batch1、无累积；普通FM epsilon-clean目标，无target time/DMD/额外损失。

## A. 编码与训练入口检查

1. 固定pilot：SELECTION的train_constant与dev_switch（若train_constant已有编码，取lexical首个未编码constant train，记录ID），两新clip分别full17/I0/prefix12，最多6次VAE encode，prefix12须与full17前缀逐值一致；text计入总上限。验证dtype/finite/尺寸、独立来源、动作长度与future隔离后直接继续。
2. 编码剩余28新clip各full17/I0，最多56次VAE encode；已编码4train复用。文本按每clip caption和独立动作句正确编码，内容相同的纯文本可按SHA复用，但不同I0/布局不得混同。全阶段最多62次VAE encode、600次text encoder调用，0采样。pilot≤900GPU秒，整个编码阶段（含pilot）≤3600GPU秒。
3. 构建训练/评测入口，零adapter在已验收b784d995固定状态与EXP-022路径比较，误差和backend记录；协议/前向不等价停止。训练eval切换及checkpoint重载使用同一adapter安装与V3wrapper；CPU先检查加载/metadata/未来截断。

## B. 128次普通FM更新

32train按seed13固定shuffle，每epoch每条恰一次，共4epochs=128updates。每次独立sigma~Uniform(0,1)和Gaussian噪声，用独立可恢复generator；当前C2的5latent为训练目标，C1为GT。每次按当前adapter重新clean commit历史，禁止跨optimizer更新复用旧KV。只训练C2；首块训练反向不在本轮，不隐含变更chunk长度/梯度范围。activation checkpoint与CPU offload保留，计入资源。

保存checkpoint0和32/64/128updates、optimizer/CPU/CUDA/数据顺序RNG与计数；8updates只做finite/有效梯度/冻结base/预算检查，通过后自动继续。记录逐步loss/sigma/数据ID/grad norm、commit/forward/backward/update耗时和显存/RSS。各阶段按固定状态probe比较，不用不同sigma的逐步loss大小证明改善。128updates是预算检查点，不是收敛保证。

## C. 固定匹配评测（训练前后相同）

- SELECTION固定2train+2dev，constant/switch各一，均checkpoint0与128updates：GT C1历史，C2 FM-8NFE，共8条GT39+生成17展示。每条1commit+8采样=9顶层DiT，共72。
- 另仅2dev在checkpoint0与128updates：从同I0各自生成C1 FM-8NFE，再clean commit并生成C2 FM-8NFE，共4条完整56帧视频，每条17顶层DiT，共68。单个checkpoint使用自己的自生成历史；前后真实历史状态不相同，这是整段生成能力比较，不能作同状态单变量误差归因。纯噪声C1的无梯度推理已有V3入口支持，不涉及首块训练反向。
- 固定4clip、sigma=.25/.75、同noise、checkpoint0/128的GT-history FM loss：每checkpoint每clip1commit+2velocity，共24顶层DiT，保存可比误差。不把该数值当视觉/动作正确率。
- 总12次视频decode（每条decode拼接后的17latent一次）；8条GT-history明确前39帧来自GT，4条self-history全部来自模型。GT展示保留原39RGB，已显示帧不被decoder覆盖。保存三列GT/checkpoint0/128片、全部新增帧图及动作时间轴、峰显存/CPU KV/每段耗时。
- 判断人物完整性、背景/ghosting、块边界与切换响应；constant/switch动作标签不能直接当正确率。训练前后的NFE、seed、初噪、动作、decoder/历史协议匹配。若开发前后都弱如实记录，不换seed或加NFE筛结果。

## 硬预算与调度

全任务累计≤13200GPU秒（3.667GPU小时）：编码≤3600、训练/额外回归及保存≤6000、全部评测≤3600，失败调用/加载计入各阶段。最多440顶层DiT（训练256+上述评测164=420，预留20回归/重载探针），128backward/128optimizer updates，62VAE encode、12VAE decode、600text调用。backward内checkpoint层重算另记可测信息并包含wall，不和顶层forward双计，不伪造未测指标。无额外自由生成或30NFE验证授权。

训练固定单GPU；编码/独立评测可最多2张实际空闲L40并行，每进程≤4CPU线程/128GiB host RAM、allocated≤44GiB；总host RAM≤256GiB，新增磁盘≤12GiB。09:00后项目总≤3卡，夜间≤8卡，本任务仍≤2卡。GPU启动前检查占用，不能抢他人卡；每进程watchdog与总共享账本限制GPU秒/调用数，卡秒按进程wall相加。

来源错误/串集/未来泄漏/前向不等价、NaN/Inf或持续无有效梯度、OOM、预算耗尽立即停止受影响任务、保存原因；不自动重试/加卡/改rank/降分辨率或扩大训练。普通loss波动不触发无依据早停；不要因8updates没有画质收益就停止128预算。若输入编码或新runner无法满足协议，先完成独立可做项并报告具体阻塞。

## 交付与决策

报告含真实执行数、编码/训练/评测独立成本、样本来源/配置SHA、训练曲线和固定probe、前后12条视频/对比与逐帧评估、checkpoint本地SHA、风险与限度。0/128updates命名清晰，FM-8NFE表示每chunk推理次数。只有GT-history变好或仅loss下降都不等于自由生成/动作通过。无新AnyFlow/DMD/SGF+/Memory。超过128updates或扩adapter容量，由Judge根据本轮结果另定，不自动启动。

开始执行、GPU启动、完成/失败都消息通知Judge。GPU预算已授权，正常阶段切换直接执行。
