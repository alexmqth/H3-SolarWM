# 一个可证伪的 ABot 动作通路 pilot（提案，未获 GPU 授权）

## 要回答的决策问题

**优先检验 H1 的一个必要环节：**在一条已有弱 A/D 分叉的 ABot 场景里，保持同一权重、真实历史、当前 noisy latent 和动作内容，仅依次关闭全局公共 prefix 的 video feedback、再切断过去 video 读取当前 video，动作条件 velocity **敏感性及其几何**是否改变？这不是语义动作正确率测试；即使速度差分很大，也只说明模型对文本有数值响应。如果 Original-visible 的差分接近重复运行误差，本状态的固定速度场探针不可判定，不能反推因果化无影响。H2“旧历史状态压制新动作”不因 H1 代理指标未过门槛便自动成立；它需要另一个独立的同状态来源×切换设计。本 pilot 不搭建 Memory/SGF+。

选择 `b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56`，因为同一原始 ABot 初图在[EXP-014 的自生成历史 A/D 视频](../../experiments/EXP-014_v3_multiscene_teacher/RECOVERY_V2_GPU_REPORT.md)中几乎同向（中央水平 flow +66.178/+58.664，生成17帧像素MAD 3.188）。这只是**选取失败表型**；本 pilot 用真实 GT C1，历史来源不同，不能将结果直接解释为 EXP-014 失败的唯一原因。A/D 是合成动作干预，不是录屏的真实动作标签。真实录屏的[已编码56帧及来源记录](../../experiments/EXP-023_v3_fm_small_data/encoded/b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56/record.json)已在本地；[EXP-014 固定条件](../../experiments/EXP-014_v3_multiscene_teacher/config.json)也指向同一源窗口。启动前必须在 CPU 核对二者的首帧/裁剪/提示词/动作布局和 source SHA；若不一致则**停下并报告**，不临时更换样本、I0 或文本。

## 固定状态与三种图

- 冻结 Original H3 + released Action LoRA，**不加载 EXP-023/025/027 的新增 FM adapter**，0 optimizer。Single I0、相同 prompt/audio、native timestep、Global RoPE、`[12,5]` 时间布局、相同 SDPA/GEMM 数值政策。沿用[已审计的固定动作位置](../../experiments/EXP-017_v3_fixed_action_layout/judge/FINAL_REVIEW.md)，只把当前 C2 的5个 action spans 全部设为 A 或全部设为 D；历史动作与所有非动作条件逐值一致。未来 C3+ 的 action/video 在 refiner 前物理裁剪。
- `H` 是该片真实视频编码的 GT C1 12 latent；当前干净 `z0` 是**同一视频** C2 的5 latent。以固定 CPU seed113 的同一噪声 `ε` 生成 `z_σ=(1−σ)z0+σε`，`σ∈{0.25,0.75}`。A/D、三种图共享各 sigma 的**同一** `H,z_σ,ε`；对照仅改变当前 action 与下述声明的 mask 边。使用 GT future latent 构造诊断状态不表示生成器推理时能看未来，它只用于固定输入的速度场探针，模型可见输入仍是 `z_σ`，没有干净 C2 泄漏。
- **R1 Original-visible：**同一可见 `[C1,C2]` 窗口内保持 Original Single-Egress 和双向 video/video，以及公共 prefix→video feedback。它不是含未知未来的完整 Original 推理。
- **R1-P：**仅去掉**全局共享的非 action 公共 prefix** query→video key；action 的 own-span 双向连接与 video/video 双向不变。R1→R1-P 测该全局反馈边组的影响。
- **R2：**在 R1-P 上仅禁止 C1 video query 读取 C2 video key；C2 可读 C1，块内仍可双向。R1-P→R2 测跨块 video 因果化的固定状态敏感性。历史在每次前向重算，**本轮不引入 persistent KV/clean-commit 时间冻结**；旧路由/KV审计已在匹配数值图下完成这一执行对照。R2 只有通过下面的多层无泄漏证明后才称“严格因果”。它是 H1 诊断图，不是新的高效模型，也**不是冻结 V3 使用的 current-prefix feedback 图**：V3 的前缀按当前块重建并有自己的可见性契约，本 pilot 的 R1-P/R2 则故意固定全局公共前缀以隔离边组。不能把 R1-P→R2 的结果直接当 V3 current-prefix 的收益或损失。

必须以实际832×480 token布局在 CPU 证明 R1/R1-P 只差公共prefix行、R1-P/R2 只差 C1→C2 的video读取边；A_i↔V_i Own Single-Egress、当前动作到当前视频、C1 的所有动作和Global位置都不变。进一步对实际 mask 做**跨50层的布尔可达性审计**：把 action、公共/其他 prefix、历史/当前 video 行和残差边全部纳入，验证 R2 中从 C2 当前信息到 C1 video、C1 action及可再被C1读取的公共/其他prefix表示均无路径；不能只查直接video边。动作文本 refiner 前物理截断未来 spans，并用两种未来文本确认可见输入逐值相同。若存在 `C2→prefix/action→C1` 的多层绕路，R2不得称严格因果，先修图并重审，不能用GPU差分解释H1。复用[前次路由审计的数值约束与代码](../../reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)，不能直接调用曾忽略 action 参数的旧 `recompute_forward()` 并假定可比。若数值/布局回归失败，止于实现问题，不解释模型机制。

## 调用、读数、可证伪结论

主实验 `3图 × 2动作 × 2σ = 12` 次顶层33B forward；在**σ=0.25和0.75各自**将 R1 的 A、D 再各运行一次校准数值噪声（新增4次），总计**最多16次**。每个调用记录权重/输入/动作/历史 SHA、可见行、sigma、速度张量摘要与完整 wall/峰显存。无更新、无VAE decode、无新视频；若现有 A/D 文本 embedding 可复用且来源一致则0 text encode，否则最多2次文本编码，必须计入账本。任何调用失败也计费。

对每个图和 sigma 分别计算 `Δ=v(A)−v(D)`、`||Δ||/(0.5(||vA||+||vD||))`、与 R1 `Δ` 的 cosine、范数比、相对L2及**该sigma自己的重复误差**。重复误差取 R1 A/D 两次输出差的最大RMS，并分别与该sigma的 `||Δ||` 和跨图差额比较。若动作差分与数值误差同量级，标为不可判定，不拿近零差分 cosine 排名；“大于误差10倍”只是数值可靠性检查，**不是 Original 语义动作正控**。范数变小表示敏感性减弱；低cosine表示敏感性方向改变，两者必须分列，低cosine不能自动解释成控制变差。`0.5`范数比或cosine仅作为探索性排序标记，不足以确立/排除机制，任何较小但重复、可能改变行为的效应仍需报告。

| 结果 | 下一判断 |
|---|---|
| R1差分接近该sigma重复误差 | 此状态的数值敏感性探针不可判定；不能说Original语义无控制，也不能说causalization无影响。停止当前pilot的机制归因，由Judge决定是否需要有行为正控的新状态。 |
| R1→R1-P 在两sigma出现可复现的范数或方向变化 | 全局公共反馈是**数值敏感性候选路径**；先问变化是否对应真实动作行为，再研究无未来泄漏的可用替代，不能直接在严格缓存里打开原全局反馈。 |
| R1-P→R2 在两sigma出现可复现的范数或方向变化 | 跨块video拓扑是候选路径；值得由Judge考虑独立episode/seed和实际17帧行为验证，不能说“KV冻结导致”。 |
| 三图都有非零差分 | 仍比较各自幅度、方向和后续行为；非零不排除H1。若各图几何在校准误差内接近，只能说本GT状态未检出该边组效应，不能外推到其他状态/生成历史。 |
| 只一档sigma改变、差分接近误差或与现有视频表型冲突 | 标记不确定；先核对数值/条件，再由Judge决定是否需要行为验证或更换研究问题。 |

**pilot不会产生动作方向或人物质量PASS。** 后续若数值干预有稳定信号，另拟同历史 A/D 两条当前17帧的匹配30NFE视频与人物/相机分离观察；若没有，也只能说此状态/边组的固定速度场证据不足，不能以之否定所有H1。要推进H2，必须提出剩余的具体行为问题及各历史来源内部的继续/切换配对，不能因为三图“都有响应”就自动跳过去。正式论文证据还需未用来选方向的 episode、多个seed、充分训练且预算匹配的因果基线，以及 Original H3 参考；本轮单状态只作是否值得进入那一步的分流。

## 预算草案与停止条件（需 Judge 另行批准）

单张空闲 L40，**最多16顶层DiT尝试（12主对照+每个sigma各2次R1重复）、2 text encode、0 backward/update、0 VAE encode/decode、0视频，GPU进程秒≤1000，allocated峰<40GiB，RSS<96GiB**。CPU预检≤4 workers/RSS32GiB，冻结全部源码/输入SHA、直接mask边集和50层可达图；在裁剪前分别填两种未来动作文本，必须证明裁剪后的可见packed tensor逐值相同，作为未来动作泄漏负控。model load、预检/重复与释放均计时。生产默认和正式V3不改；新入口放独立目录，不覆盖旧审计。若输入来源/固定布局不匹配、多层无泄漏证明失败、数值执行回归失败、资源触限或出现未声明的mask边变化，停止并保留真实负结果，不自动换场景或追加视频。若R1差分弱，保留16次上限中的未用调用，不把它们改用于临时换场景。这个预算是**提案，不是运行许可**。

本 pilot 比重复停车场72次路由审计新增的信息，是在**已有真实ABot弱分叉场景**、真实GT C1/C2状态上测试同一边组效应能否出现，并先检查 Original-visible 是否本来也弱。它仍无法单独解释自生成历史下的失败；H2 需要随后在各自历史来源内部做配对动作、观察来源×动作切换交互，不能把GT与self的不同轨迹直接相减。
