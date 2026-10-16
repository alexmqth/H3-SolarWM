# 一个可证伪的 ABot 动作通路 pilot（提案，未获 GPU 授权）

## 要回答的决策问题

**优先检验 H1：**在一条已有弱 A/D 分叉的 ABot 场景里，保持同一权重、真实历史、当前 noisy latent 和动作内容，仅依次关闭公共 prefix 的 video feedback、再切断过去 video 读取当前 video，动作条件 velocity 差分是否在某个边组上稳定塌缩？如果 Original-visible 正控本身也弱，应停止把该场景的弱动作归因于因果化。H2“旧历史状态压制新动作”留待 H1 排除后另立任务，不在本 pilot 中搭建 Memory/SGF+。

选择 `b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56`，因为同一原始 ABot 初图在[EXP-014 的自生成历史 A/D 视频](../../experiments/EXP-014_v3_multiscene_teacher/RECOVERY_V2_GPU_REPORT.md)中几乎同向（中央水平 flow +66.178/+58.664，生成17帧像素MAD 3.188）。这只是**选取失败表型**；本 pilot 用真实 GT C1，历史来源不同，不能将结果直接解释为 EXP-014 失败的唯一原因。A/D 是合成动作干预，不是录屏的真实动作标签。真实录屏的[已编码56帧及来源记录](../../experiments/EXP-023_v3_fm_small_data/encoded/b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56/record.json)已在本地；[EXP-014 固定条件](../../experiments/EXP-014_v3_multiscene_teacher/config.json)也指向同一源窗口。启动前必须在 CPU 核对二者的首帧/裁剪/提示词/动作布局和 source SHA；若不一致则**停下并报告**，不临时更换样本、I0 或文本。

## 固定状态与三种图

- 冻结 Original H3 + released Action LoRA，**不加载 EXP-023/025/027 的新增 FM adapter**，0 optimizer。Single I0、相同 prompt/audio、native timestep、Global RoPE、`[12,5]` 时间布局、相同 SDPA/GEMM 数值政策。沿用[已审计的固定动作位置](../../experiments/EXP-017_v3_fixed_action_layout/judge/FINAL_REVIEW.md)，只把当前 C2 的5个 action spans 全部设为 A 或全部设为 D；历史动作与所有非动作条件逐值一致。未来 C3+ 的 action/video 在 refiner 前物理裁剪。
- `H` 是该片真实视频编码的 GT C1 12 latent；当前干净 `z0` 是**同一视频** C2 的5 latent。以固定 CPU seed113 的同一噪声 `ε` 生成 `z_σ=(1−σ)z0+σε`，`σ∈{0.25,0.75}`。A/D、三种图共享各 sigma 的**同一** `H,z_σ,ε`；对照仅改变当前 action 与下述声明的 mask 边。使用 GT future latent 构造诊断状态不表示生成器推理时能看未来，它只用于固定输入的速度场探针，模型可见输入仍是 `z_σ`，没有干净 C2 泄漏。
- **R1 Original-visible：**同一可见 `[C1,C2]` 窗口内保持 Original Single-Egress 和双向 video/video，以及公共 prefix→video feedback。它不是含未知未来的完整 Original 推理。
- **R1-P：**仅去掉非 action 公共 prefix query→video key；action 的 own-span 双向连接与 video/video 双向不变。R1→R1-P 测公共反馈影响。
- **R2：**在 R1-P 上仅禁止 C1 video query 读取 C2 video key；C2 可读 C1，块内仍可双向。R1-P→R2 测严格跨块 video causality。历史在每次前向重算，**本轮不引入 persistent KV/clean-commit 时间冻结**；旧路由/KV审计已在匹配数值图下完成这一步执行正确性检查。本轮的 R2 是 H1 诊断，不是新的高效模型。

必须以实际832×480 token布局在 CPU 证明 R1/R1-P 只差公共prefix行、R1-P/R2 只差 C1→C2 的video读取边；A_i↔V_i Own Single-Egress、当前动作到当前视频、C1 的所有动作和Global位置都不变。复用[前次路由审计的数值约束与代码](../../reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)，不能直接调用曾忽略 action 参数的旧 `recompute_forward()` 并假定可比。若数值/布局回归失败，止于实现问题，不解释模型机制。

## 调用、读数、可证伪结论

主实验 `3图 × 2动作 × 2σ = 12` 次顶层33B forward；另在σ=0.25将 R1 的 A、D 各**重复一次**校准数值噪声（新增2次），总计**最多14次**。每个调用记录权重/输入/动作/历史 SHA、可见行、sigma、速度张量摘要与完整 wall/峰显存。无更新、无VAE decode、无新视频；若现有 A/D 文本 embedding 可复用且来源一致则0 text encode，否则最多2次文本编码，必须计入账本。任何调用失败也计费。

对每个图和 sigma 计算 `Δ=v(A)−v(D)`、`||Δ||/(0.5(||vA||+||vD||))`、与 R1 `Δ` 的 cosine/相对L2及重复前向误差。先要求 R1 的动作差分范数超过重复数值误差10倍，且在两个 sigma 都非退化；否则 **H1在该状态不可判定**，不拿近零差分 cosine 排名。预注册 pilot 的“明显损失”读数为：两个sigma均有 `||Δ_variant||/||Δ_R1||<0.5` **或** `cos(Δ_variant,Δ_R1)<0.5`，同时差额超过重复误差10倍。这只是选择后续行为验证方向的门槛，不是动作语义/统计显著性结论。

| 结果 | 下一判断 |
|---|---|
| R1弱或重复噪声与动作差分同量级 | 当前ABot状态无有效Original-visible正控；停止H1归因，换“此动作/状态本身为何弱”的问题，不能用本pilot宣称causal退化。 |
| R1强，R1-P明显弱，R2没有新增下降 | 公共prefix feedback 是此状态的候选关键通路；下一步要设计**无未来泄漏**的等价可用条件路径，并做行为视频，不能直接在严格缓存里打开原全局反馈。 |
| R1与R1-P强，R2明显弱 | 跨块video双向回路在此状态承载动作影响；H1值得在独立episode/seed与实际17帧行为上复核，不能说“KV冻结导致”。 |
| R1、R1-P、R2都强 | 此GT-history状态不支持H1，才考虑同图下的 self/GT history × 继续/切换动作交互来检验 H2。 |
| 只一档sigma改变、方向矛盾或视频先验证据冲突 | 标记不确定，不用单个cosine写机制论文故事；先核对数值/条件，再由Judge决定是否需要行为验证。 |

**pilot不会产生动作方向或人物质量PASS。** 后续若数值干预有稳定信号，另拟同历史 A/D 两条当前17帧的匹配30NFE视频与人物/相机分离观察；若没有，禁止通过增LoRA或调anchor救机制假设。正式论文证据还需未用来选方向的 episode、多个seed、充分训练且预算匹配的因果基线，以及 Original H3 参考；本轮单状态只作是否值得进入那一步的分流。

## 预算草案与停止条件（需 Judge 另行批准）

单张空闲 L40，**最多14顶层DiT尝试、2 text encode、0 backward/update、0 VAE encode/decode、0视频，GPU进程秒≤900，allocated峰<40GiB，RSS<96GiB**。CPU预检≤4 workers/RSS32GiB，冻结全部源码/输入SHA、mask边集；在裁剪前分别填两种未来动作文本，必须证明裁剪后的可见packed tensor逐值相同，作为未来动作泄漏负控。model load、预检/重复与释放均计时。生产默认和正式V3不改；新入口放独立目录，不覆盖旧审计。若输入来源/固定布局不匹配、Original-visible正控弱、数值执行回归失败、资源触限或出现未声明的mask边变化，停止并保留真实负结果，不自动换场景或追加视频。这个预算是**提案，不是运行许可**。

本 pilot 比重复停车场72次路由审计新增的信息，是在**已有真实ABot弱分叉场景**、真实GT C1/C2状态上测试同一边组效应能否出现，并先检查 Original-visible 是否本来也弱。它仍无法单独解释自生成历史下的失败；H2 需要随后在各自历史来源内部做配对动作、观察来源×动作切换交互，不能把GT与self的不同轨迹直接相减。
