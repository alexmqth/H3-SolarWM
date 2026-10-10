# EXP-007 / v3 — V3-AF有限训练与匹配8NFE视频验证

2026-10-11 02:05 HKT，Judge。AF1真实warmup通过，**AF2训练已完成并经Judge接受工程结果；AF3视频阶段已批准执行，见文末放行记录**。用户夜间连续授权至09:00，使用实际空闲GPU，建议GPU1；09:00后最多3卡。训练和评估分别预算；AF3已由Judge放行，不要求用户重批。

## 研究问题与基线

Research Track C；Parent为冻结V3 causal/KV协议上的新AF1 student。验证32次以内finite-map更新后，8NFE能否保留可辨A/D响应、人物与场景，是否具有继续AnyFlow/DMD的研究价值。普通FM8的EXP-006已经可行；本实验只训练一个配置，不扫参。训练loop通过不等于生成能力PASS。

## 控制与训练输入

Original H3 + released Action LoRA冻结；Single I0/native video1000sigma/audio1000，current-prefix feedback、own-action/action feedback、Global、strict causal/max_history5、首12后5、h3_fp32边界与现有backend不变。新增target-time MLP gate.25和last8 rank8 QKV延续AF1，不加载旧AnyFlow权重。

从H3-World/outputs/EXP-007_v3_anyflow_af1_attempt2/step_01恢复配套target_time.pt/qkv.pt/trainer_state.pt。恢复optimizer与logical/CPU/CUDA RNG；必须在模块安装后恢复RNG。配对SHA见AF1结果与judge/af1_audit.json。metadata一致性拒绝混用step或旧anchor协议；新增V3显式校验，不能借旧validator名义绕过协议核对。

冻结V3 C1为submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt；AA/AD C2分别为H3-World/outputs/EXP-002_native_cached/{AA,AD}/chunk_12_17.pt。偶数累计step用AD，奇数用AA。每logical batch4仍2FM+1endpoint+1general finite-map，4次backward、一次update。AdamW lr1e-4/betas(.9,.95)/wd.01/clip1、shift2.22/epsilon5，训练source是generated V3数据，不是GT或泛化集。

## AF2执行、预算与停止

开始前冻结runner/config/input/source/hash及恢复状态；必要CPU preflight。每次更新前用**当前student权重**重建一次C1 sigma=r=0 detached raw KV。batch内复用，更新后释放，下一step重建。记录动作prompt与endpoint配对、cache身份、finite输出/梯度、实际calls/latency/memory。不得跨不同权重沿用KV。

**新增最多31updates（累计step32）；527forward=31×(1prefill+16training)，124backward，31optimizer update，0VAE，≤2.0 GPU小时，allocated peak≤44GiB，CPU4threads。** 失败calls计入；不得自动重试或重置账本/peak。绝对截止min(阶段2小时,09:00)。根据AF1约183秒/次含初始化与额外诊断，此预算允许有限训练且无需扩大配置。

只保存step8、step32及若到预算边界时的最后完整update，含配套权重/optimizer/RNG/源码与协议metadata。不逐步保存大KV、不复制base weights。按实测每步时间预留保存时间，在预算内完整update边界收口；时间不足允许少于32steps，不能伪称全部完成。OOM/nonfinite/基础权重变化/cache协议错误立即停止，提交证据交Judge，不盲目重跑。

## AF3预定义评估（已于03:21由Judge放行）

预计≤35forward/4VAE/0update/0.35GPU小时；使用最终预定checkpoint，不扫多个checkpoint挑片。EXP-006新FM8 C1为共同clean历史，各模型按自己的权重构建KV；使用seed13冻结C2/C3 initial_noise slice、同audio/I0/actions/Global/native shift2.22九sigma点。普通FM8 AA/AD C2/C3原结果可复用。AF两条C2共享一次C1 prefill，16sampling；C3为两条各自AF历史，2clean commit+16sampling。合计35forward，4decode。

每次map为x_r=x_t+(r-t)*v(x_t,t,r)，r取下一个sigma；8NFE匹配FM8。前39RGB借用FM8并保持不变，明确称AF continuation，不称从首窗全程AF。C2是同clean历史比较；C3是各自生成历史比较。FM8与训练AF为目标条件+训练联合方案，不宣称纯训练单因素效应或独立泛化。

## 验收、交付与ROI

查看全部新增帧和代表视频：A/D方向与切换、主体结构、ghosting、场景与chunk boundary、历史RGB不变。普通缺陷允许PARTIAL；严重持续主体/控制失败则停止该配置，不要求大量训练挽救。有限可行但无已证实FM8收益也如实收口。

保存训练逐步指标/原日志、calls与资源账本、源/输入/checkpoint清单、配对配置和恢复信息。视频阶段保存原片、并排对比、辅助flow及逐块sampling/prefill/commit/decode/memory成本。小证据归档到本实验目录，权重保持outputs；Worker更新report，Judge审核后同步mainline/report与Git。完整V3 Baseline冻结不覆盖。

## 前置结论与下一步边界

AF1工程PASS：真实初始化差异0，target/QKV有效梯度和更新、基础权重不变、更新后KV重建有效；两attempt累计186.58153秒/22forward/4backward/1update/0VAE。详见judge/AF1_REVIEW.md。

训练后先做视频，依据实际能力决定后续DMD。可在AF2运行时准备FUTURE_DMD.md所述CPU角色/符号/真实多步梯度验证；本任务没有DMD GPU授权，不改变当前AF2。训练结束不能因loss下降直接宣布AnyFlow成功。

## Judge阶段放行 — 2026-10-11 03:21 HKT

AF2已完成并接受工程结果：累计step32，新增31updates/527forward/124backward/0VAE/4287.86071秒，peak26.76744GiB，进程已退出。配对checkpoint及完整账本独立审计通过，见submission/experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md。

**现在批准执行AF3全部预定义视频阶段**，使用AF2最终step32：shared C1 prefill→AA/AD C2→AA/AD C3，共35forward=1prefill+2commit+32sampling、4VAE、0update、≤0.35GPU小时、allocated≤44GiB、最晚09:00。建议实际空闲GPU0；不覆盖原片。CPU preflight先冻结源码/config/输入/配对checkpoint，通过后直接启动，无需用户再次批准。准许marker为实验目录AF3_GPU_AUTHORIZATION.json。异常停止交Judge，无自动重试，不扩到其他checkpoint或NFE。

C1为EXP-006 FM8共同clean历史与39RGB；AF用自身权重prefill KV。C2同可见历史比较、C3各自生成历史。保存完整AA/AD73原片与匹配对比、逐块耗时/显存/缓存与RGB不变性。训练C1来自原30步V3，评估C1来自FM8，这一历史分布差别如实记录。训练结果不直接构成生成能力验收。

## Judge修复重试放行 — 2026-10-11 03:26 HKT

AF3首次prefill因缺少冻结runtime/code/causal导入路径，在模型forward前退出：0forward/0VAE，3.560679秒已记入原账本。修复路径、补充CPU真实导入预检及依赖摘要后，Judge独立CPU_PASS（step32、18项source、native8step）。批准一次修复后attempt2，沿用同输出和累计账本；原35forward/4VAE/0.35GPU小时/44GiB及09:00截止不变。保留attempt1日志与manifest，后续异常仍停止交Judge。
