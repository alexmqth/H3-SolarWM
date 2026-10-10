# EXP-012/v1 — 冻结 AF8 在共同 FM8 历史上的短程场景迁移

2026-10-11，Judge。Research Track：Branch C / AnyFlow evaluation。**当前只授权P0 CPU准备；GPU必须经Judge按场景单独发放冻结marker。** 依据用户持续研究授权，不需用户重复确认。此任务不启动新训练。

## Parent / 研究问题 / 假设

Parent为EXP-007 AF2 step32的V3 target-time student；对照为EXP-011工业/村落普通FM8。停车场有限训练是否在两个固定其他场景上保留匹配8NFE续写能力，是否带来清楚可见的联合收益？假设允许被否定，不预设AF优于FM。

决策价值：若AF8在相同历史下明显退化或没有稳定收益，保留普通FM8作为低成本候选，冻结当前AF8并把多样化数据训练列为后续独立计划；不再做checkpoint/seed/步数微调来救当前配置。若有收益，只支持有限范围的新证据，不宣称成熟泛化。

## Inputs / 唯一 checkpoint

复用EXP-011已验收的两个native Single I0 fixture，以及各自 **G1/FM8** first12.pt / published_39.npy；AA/AD的C2初噪声和prompt仍来自同一fixture。对照原片/指标直接复用EXP-011 G2/FM8，不重跑FM8。

- industrial fixture SHA：9f404c910c31e68b319cf6029d7d59fa48229d9db14df395a2890cc32057107d。
- village fixture SHA：eb7590213fb7f3971f3e6d345bce185ab8f9b6fba757f43a2123b65631478775。
- AF checkpoint目录：H3-World/outputs/EXP-007_v3_anyflow_af2/step_32/。
- qkv.pt SHA：07c8e5e68d1c947e60217e326ca8dd4c00222a555542b286dfc03514e27e916e。
- target_time.pt SHA：ceb7d62324834a5b2be9fd6a47bfe4b5a9d810e49d1bb33889af0fb788940f70。
- 配套元数据：累计step32、last8 blocks42–49、rank8、scale0.125、target gate0.25，遵循EXP-007 AF3已验收加载协议。

P0逐项核对真实权重与配对元数据、两个C1的端点/RGB SHA及各自fixture路径；写入独立source manifest。训练来源仍为有限停车场FM30端点，本轮初图来自已有游戏录屏validation episode，不称绝对未见或现实摄影数据。

## Controlled / Changed Variables

保持Original H3+released Action LoRA、Single I0、native full37 packed布局与Global位置、own-action routing、action feedback、current non-action prefix、audio1000、strict causal、clean persistent CPU raw KV(max_history5)、12+5 latent、832×480/24fps、seed13和初始video/audio噪声不变。

改变模型方案：加载冻结target-time/QKV并采用已验收AnyFlow finite-map一步映射，相邻native8 sigma分别作为source/target；不能称纯训练单变量。普通FM8保持已保存结果。

每场景两个模型共享C1 **latent/RGB**；AF必须用自身权重重建C1 raw KV，不读取FM8缓存给student使用。AF clean commit source_sigma=target_sigma=0，无梯度。AA/AD共用该AF cache及C2初噪声，仅C2 action spans变更。历史prefix/time/action注意力不能暗改。

## P0 CPU实现与核查

Worker在本独立目录实现runner/config/manifest，优先复用EXP-007 interval_student及EXP-011输入/账本工具，禁止修改冻结父实验代码或结果。CPU验证真实入口导入、权重配对、fixture与C1来源、完整position/prompt可见性、native8/target-time网格、clean commit语义、预算和截止闸门；检查AA/AD只改当前action rows。拒绝无marker启动，拒绝覆盖输出或自动重试。模拟模型仅验证协议/记账，不能作真实模型能力证据。

所有GPU实际调用预记账，loader启动/失败耗时也累计。清楚记录环境、FP32权重恢复/输入精度、attention backend、ABOT_VRAM_RESERVE_GIB=18与原模型路径/offline设置。实际运行前重新核查物理GPU空闲，不碰GPU3/4的他人进程。

## G1 有限GPU阶段（P0审核后逐场景放行）

每场景：加载唯一student，从共同FM8 C1做1次clean commit；AA/AD分别8次finite-map、各1次decode，生成17新帧，最终两片56帧。先工业后村落；各场景独立marker与证据。前一场景存在持续结构失败时Judge决定是否保留另一预定场景的有限机会，Worker不自行扩大或重试。

总上限：**34 denoiser forward=32sampling+2commit，4decode，0text/image encoder，0backward/update，≤1260 GPU秒(0.35 GPUh)**。每卡allocated≤44GiB；空闲磁盘≥60GiB，新cache/tensor预计≤16GiB；09:00 HKT绝对截止。使用实时空闲GPU，项目夜间无额外总卡数限制；09:00后项目≤3卡且本任务不自动续跑。预算不因启动失败重置。不可挪用余额加C3、首窗、seed或新场景。

## Evaluation / Acceptance

逐场景/分支检查全部34新帧、边界与原分辨率代表帧；AA/AD动作方向/切换、人物/场景结构、ghosting和边界分别评估。沿用EXP-011中央ROI每转移中值水平流16次求和与RGB MAD作辅助，不能代替语义判断或称控制准确率。

独立核查：student own-weight raw KV的50层index0/bytes及来源、采样不修改历史cache、共享C1/noise、8次source/target sigma、旧39RGB逐值不变、完整56帧24fps解码/PTS。每场景报告加载/commit/sampling/decode/保存耗时，allocated GPU和CPU KV内存，原始配置、调用账本、日志、端点/cache/video SHA和代表视频。

与FM8的比较在共同C1、相同动作/noise与8NFE下进行，但两模型KV由自身权重形成。标为matched-history model comparison；不是相同raw KV的消融，也不是从首窗全程AF。普通视觉缺陷PARTIAL；持续人物分解/全场景彩噪/主体长期消失为关键失败。

## Stop / Out of Scope

OOM/nonfinite、来源/协议/权重错误、超预算或截止/磁盘阈值，立即停止后续启动并报告；不自动重试。视觉持续崩坏停止该配置，不通过换checkpoint/种子/prompt挽救。无C3、无AF首窗、无长视频/SW、无FM4/AF4/DMD、无训练和新LoRA、无AnyFlow超参微调、无额外场景。

## Deliverables / Closure

完整Worker report、CPU结果、冻结config/source/code manifests、分阶段marker、实际调用账本与命令日志、四条56原片/四段17新帧与全帧图、每场景AA/AD两条FM8-vs-AF8比较片、指标/限制、Judge审核。证据归EXP-012，展示归report/v3/v3_anyflow；原大tensor留本机，不提交Git。最终由Judge更新progress/next_plan/archive并正常发布GitHub，不覆盖任何正式V3结果。

## P0审核与工业GPU放行（2026-10-11 06:55 HKT）

Judge独立CPU预检与预算/finite-map/marker测试PASS，最终代码清单4a8e4646…db1e4、来源95bbdf91…5f7d已冻结。批准GPU0工业一次17forward/2decode，保持总1260秒与09:00截止；marker见EXP-012/judge/G1_industrial_APPROVED.json。村落待工业证据后单独放行，不重试/不扩展。
