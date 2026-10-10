# 后续独立研究计划：V3-FM8 与 V3-AF

**设计提案，未授权执行。EXP-005不训练、不安装新adapter、不加载旧AnyFlow checkpoint作为结果。** 后续各任务需独立EXP编号、冻结协议与资源预算。

## 1. 两个问题分别回答

| 候选 | 改动 | 所回答的问题 | 当前证据 |
| --- | --- | --- | --- |
| V3-FM8 | 原V3权重，普通native FM/Euler每新块8步 | 原权重直接减步是否足够 | EXP-004仅复用30步首窗后续到73帧；全程8步未测 |
| V3-AF | 新target-time-conditioned student，训练finite-map目标 | 训练后能否在匹配8 NFE下改善有限步长映射 | 当前V3协议没有已验收训练结果 |

先在明确冻结的V3 Baseline Global/KV协议上做实验；SW候选未验收时不把Local或eviction悄悄加进少步训练。若后续选择SW-G作为部署协议，须单独锁定window/position/cache条件并重新列对照。

## 2. V3-FM8先解除首窗依赖

建议单场景/固定seed，首12用8步从原始I0生成，然后AA/AD各续两个5latent块；与同协议30步使用相同起始噪声。先检查首39RGB可用，再分叉当前A/D，最后续自身历史。普通缺陷可接受，严重结构崩坏则停止，不扫描4/12/16steps或shift。

未来单独任务的初始建议上限：首窗8+4新块×8=40sampling，共享首窗commit1+两个第二块commit2，共43forward、5VAE、0训练、≤0.75GPU小时、1卡。30步参考优先已有材料，若无法匹配首窗输入则标不匹配并另批参考成本。该预算**不属于EXP-005，也未获批准**。

## 3. V3-AF的student、time与目标

1. 从已验收V3 Original H3 + released Action LoRA建立独立student，base/released LoRA冻结；新建target-time MLP与有限rank QKV LoRA作为可训练参数，保存新的checkpoint身份。旧V1/V2a/旧anchor/旧time下的AnyFlow产物只作历史资料，不加载冒充新student。
2. 维持当前chunk读取已提交历史raw KV、原own-action/action feedback/current non-action prefix feedback、Single I0、Global RoPE、native音频/文本条件。训练新增target-time条件，仅在明示模块注入；不另开action gain/mask/prefix修复。
3. **以冻结V3实际入口为准：** video timestep=`1000*sigma`、audio=`1000`、`fixed_prefix_timesteps=False`、clean commit=`sigma0`。SolarWM参考使用native data-ward `1-sigma`转换；本仓旧AnyFlow代码也含另一time标记，不能直接复制其转换。先以CPU解析模型/调度器及小型线性场检查，确认速度符号与时间单位，再锁定adapter。
4. 采用noise-ward表述：`x_sigma=(1-sigma)*x0+sigma*epsilon`；student输出区间平均场`u_theta(x_sigma,sigma,r,history,conditions)`，finite map为`x_r=x_sigma+(r-sigma)*u_theta`，其中`0<=r<=sigma<=1`。对角`r=sigma`退化为普通FM velocity；终点`r=0`为一步区间映射。普通Euler调用不具备这个目标时间能力。
5. target-time MLP从当前实际加载的time MLP复制，明示固定混合门（例如0.25），初始化时`r=sigma`应恢复原time embedding和forward，prefix/time row选择也须逐项验证；零初始化QKV增量、adapter dropout=0，不改变噪声RNG。保存新target模块与LoRA配对metadata，不能混不同step。
6. 参考AnyFlow v1.5的有限差分目标：同一history/conditions下，3次detached调用得到对角场与bounded plus/minus轨迹差分，再1次有梯度prediction；残差为`u+(t-r)*D_t u-(epsilon-x0)`，需统一raw1000时间与归一化导数的单位。保留diffusion/endpoint/general-map采样混合，先测r=t一致性与解析toy场；不能仅loss下降宣布通过。

## 4. 数据与cache不能混用

先复用已验收V3的generated clean latents作teacher-forced训练来源，明确它们不是GT、多场景或独立泛化验证。训练/验证按已冻结的样本/噪声条件拆开，不将训练端点直接当独立测试。数据不足以泛化就称单场景适配可行性。

teacher与student参数不同，**不能直接把冻结teacher raw KV当成student自己计算的历史KV**。每个优化步骤用当前student与同一clean history在同一sigma0协议下构建detached历史cache，在该步4次AnyFlow调用间固定；参数更新后刷新，禁止跨optimizer step复用旧权重的cache。首轮可只训练第二块，仅需共同首窗cache，控制成本。训练前后完整缓存来源、权重版本和时间条件可追溯。普通inference的no_grad保护需要独立gradient-read入口，不能修改Baseline入口以偷偷放开训练。

后续生成评估时每个模型用自己的权重和自己的生成历史commit KV。固定history对照可共享clean latents/输入噪声，但各模型须自己计算cache；不能要求不同权重的raw KV字节相同。

## 5. 预算与匹配对照设计（未来单独审批）

- AF0：CPU公式/target-time初始化/gradient routing与cache provenance验证，GPU0。
- AF1：建议最多32 optimizer updates，logical batch4（2 diffusion +1 endpoint +1 general-map），最多512个AnyFlow训练forward，最多32个共享首窗cache prefill；初步单卡≤4GPU小时。每次forward/backward、cache构建、失败都记账。只在资源可承受且完整任务获批后启动，不根据loss自动扩至128/更多步。
- AF1评估建议AA/AD第二/第三块各8 NFE，共32sampling +模型自己的首窗prefill/第二块commit最多3，共35forward、4VAE、≤0.75GPU小时；与V3-FM8固定同噪声、同history输入、同action、同video位置/缓存协议比较。各自产生历史第三块属于闭环；同状态动作证据单独看第二块。
- 以匹配**sampling NFE8 vs8**为主，30 NFE基线仅作质量上限参考。端到端账本必须另列prefill/commit/decoder、首屏、完整生成、实际显存/耗时；训练总成本也公布，不能仅用8比30宣称方法收益。
- 如果需要归因于finite-map目标而非普通额外训练，另立匹配trainable参数、训练样本、optimizer updates及总计算量的ordinary-FM适配对照；不把该额外训练默认为当前预算。先有足以影响决策的信号再做归因对照。

停止：初始化不能恢复对角/时间单位或梯度路径有误则不启动训练；nonfinite、预算用尽、核心动作/结构持续退化即停；完成32步后若无可辨收益，保存负结果，Judge决定结束或另立一次有依据的调整。当前不启动DMD/critic或on-policy循环。

## 6. 参考与限制

SolarWM checkout参考`ce1da4e7705391eda8eeda6016c0fd3f614b975e`的`anyflow_conditioning.py`、`anyflow_loss.py`、`stage1_sampling.py`及`training/anyflow.py`。其time符号、camera PRoPE、prefix、固定5latent W6与当前V3不同。旧本仓`code/causal/anyflow.py`也有明确旧协议time/adapter校验；只参考公式和测试思想，不能直接声称V3-AF已经完成。
