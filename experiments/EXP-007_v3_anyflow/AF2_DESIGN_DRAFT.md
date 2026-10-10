# AF2有限训练与匹配8NFE评估 — Judge设计草案

2026-10-11。**这是准备文档，不增加GPU授权；AF1真实warmup完成后由Judge写入next_plan并冻结任务书。**

## 研究问题与决策

在保持V3 causal/KV/current-prefix条件时，一个小预算的新target-time student能否保留可辨A/D控制和主体结构？普通FM8已可行，本轮先判断finite-map训练是否值得继续。loss变化只作训练诊断；不要求短训练改善每个画质细节。没有可辨能力或收益时归档，停止无依据扫参。

## 候选训练范围

接续AF1 step1的配套QKV、target、optimizer、CPU/CUDA/logical RNG；最多累计32updates（新增31），原lr/rank/gate/批次结构不变。偶数step用冻结V3的AD C2，奇数step用AA C2；同一冻结C1，明确是generated teacher data上的窄域适配。首次D样本必须核对动作prompt与端点配对。

每次optimizer更新后，旧student KV失效，下一次训练先用当前权重重建C1。各logical batch四样本共16forward+4backward；每新增update另1历史prefill。因此新增31步最多527forward/124backward/31update/0VAE。停止时无需多做一份最终大KV；评估另行prefill。

仅保存step8、step32（或到预算停止时已完整完成的最后step）及配套optimizer/RNG/metadata，不逐步保存大KV，不增加基础模型拷贝。预算根据AF1真实耗时确定，绝对截止09:00。NaN/OOM/冻结权重或cache协议错误立即停；预算不足则在完整update边界停止并如实报告，不暗重跑。

## 首选评估

固定EXP-006新FM8首12 latent作为各候选共同clean历史；噪声使用冻结seed13的C2/C3对应slice，音频、I0、动作、Global位置、native shift2.22九个sigma点全一致。训练noise generator与此评估noise不同，但场景和动作不是独立泛化集。

普通FM8的C2 AA/AD结果及C3各自历史结果已有，可直接复用。AF每个checkpoint必须自己prefill相同C1，不能加载FM8原始KV。target/QKV checkpoint校验step、协议、config/source hash配套；旧AnyFlow validator采用旧chunk/anchor键，不能盲目绕过或混用，应给新V3协议写显式配对检查。

AF每步使用目标r为下一个sigma的有限映射，x_r=x_t+(r-t)*v(x_t,t,r)，与FM8匹配8NFE和sigma点。两条AF C2复用同权重C1 cache，共1prefill+16sampling；C3为各自AF生成历史，另2clean commit+16sampling。共35forward/4VAE，可展示AA/AD73；前39RGB明确借用FM8。

首先评估最终固定训练checkpoint；是否需要step0或step8生成取决于明确归因问题，不能自动形成多checkpoint视频扫选。单纯比较FM8与训练AF是目标条件+训练的联合方案评估。不得把借用FM8首窗的结果称全程AF生成或完整AF首窗能力通过。

## 验收

逐块检查全部新增帧：A/D方向和切换、人物结构、ghosting、场景与边界、已发布前缀RGB保持；按可行性尺度允许PARTIAL。保存原片与匹配比较、配置/hash、source/noise/history/KV身份、采样/prefill/commit/decode/保存耗时、forward/NFE、GPU/CPU memory。每阶段及总预算完整记账。

若仅训练机制成立，标记工程PASS/生成NOT_TESTED或FAIL；若控制与主体可用但未改善普通FM8，记录有限可行性并不宣称收益。后续DMD需独立CPU角色/梯度核查及一次cycle预算，不能把旧stage2-lite产物算作新V3结果。
