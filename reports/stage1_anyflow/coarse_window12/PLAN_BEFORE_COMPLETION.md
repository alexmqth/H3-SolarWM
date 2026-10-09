# Next Plan v4：先恢复局部动作信息流

2026-10-09 02:57。此计划按用户要求替代旧 A–D 扩展路线。旧计划全文已归档；Stage1尚未通过，不继续FM48/AnyFlow136，不启动旧单步paired-velocity GPU更新。

唯一当前优先级：Original H3 + released action LoRA，在没有未来动作/视频输入的条件下，恢复首块及后续块的局部action response与人物结构。

| 实验 | 核心干预 | 决策证据 | 启动状态 |
|---|---|---|---|
| E1 局部动作信息流 | T1：过去KV冻结、prefix读取允许历史；T2：局部Original双向窗口每个sigma重算 | 无未来泄漏、后续chunk同状态A/D改善、30步局部方向正确、人物完整 | T1/T2与时域测试通过；native时间＋单I0整窗口正控恢复；5+5+2两历史仍FAIL；匹配首5latent失败、12latent首窗方向恢复；后续窗口运行 |
| E2 动作后果监督 | 同状态fork得到可靠A/D transitions；FM对照 vs FM+action-swapped transition ranking | 正确动作绝对拟合及held-out真实生成一起改善 | 仅在拓扑/正控可信而动作仍弱时启动 |
| E3 AnyFlow→Stage2 | 先减少可信causal模型的每块步数，再适应自身history | 4/8接近causal30；free rollout优于自身基线；端到端实测 | 局部动作/画面门槛通过后启动 |

**2026-10-09 04:57执行更新：** 完整12latent当前窗口的动作正控在恢复原始时间＋单首帧后通过方向检查，A−D=2.227735；这不是5latent causal通过。原生单首帧5+5+2检验已完整结束并No-Go；其他训练不前移。[校准证据](submission/reports/stage1_anyflow/local_topology/CONDITIONING_RESULTS.md)。

## E1 下一组实际工作

2026-10-09 05:32：先收齐现有GPU1/5的窗口1/2，禁止重启或加训练。匹配5latent与12latent首窗结果见[同区间报告](submission/reports/stage1_anyflow/coarse_window12/FIRST_WINDOW_RESULTS.md)；后续通过才做匹配条件的same-state geometry及GT局部视频。

当前唯一下一GPU候选为[12latent窗口有限协议](submission/reports/stage1_anyflow/local_topology/coarse_window_protocol.json)，已启动。使用现有Original124f A/D参考，在history_stop=0/12/24处各fork当前窗口A/D，保持原生时间、单I0、30steps和重算语义；两history共360次noisy forward；因37latent fixture不同，另做同输入首5latent A/D控制60次、诊断9次，零训练。先冻结37latent布局/noise并检查未来隔离、VAE prefix显示范围；不能把旧39f fixture当逐bit匹配，也不是生成124f最终demo。若后续窗口仍无可靠控制，停止粒度扩展。下列是E1总体检查清单：

1. 隔离实现T1/T2，先验证多层信息可达性、future-action/video负对照、refiner/anchor/VAE时域依赖与cache失效规则。T1每块clean commit只记录本块video KV；T2不能复用受当前sigma/action影响的旧prefix/history hidden KV。
2. 校准停车场Original-prefix局部正控：只输入已知历史与当前块，不用全长双向teacher的未来状态来定义监督。T2与该正控使用相同计算时只运行一次，不把自身delta cosine=1当效果收益。
3. 固定Original权重、39RGB/12latent/5+5+2、history5、h3_fp32、flow shift2.22、30steps/chunk、seed13。初版RGB dual与clean text time正控失败；独立单变量校准后，以原生text time＋单I0检验局部能力，保留旧协议对照。I0不得retime。参照成立后才做匹配条件的C0/C1/T1/T2同状态geometry；首块与后续分列。
4. 同一历史/同一当前noisy latent只替换当前A/D。主筛查2份共同history×3chunks×3sigma；图像/动作/噪声/位置哈希一致，完整保留失败。入选路线补seed29与切换。GT-history画面使用真实ABot；停车场Original-generated history单独标注，不能称GT。
5. 仅在局部方向与画面都可靠后，选择persistent-KV或窗口重算路线；后者允许成为主原型，但必须报告真实重算成本。不要求先完成20秒稳定生成。

## Go / No-Go

- **局部动作：** 无未来泄漏；多个chunk的同状态action delta与实际运动都可信，仅首块提升不通过。
- **局部画面：** GT history + 30steps下人物/场景没有严重分解；oracle重置不能冒充自由生成。
- **Few-step：** 可信causal30建立以后再验4/8；不以finite residual下降代替画质与动作。
- **Generated history：** 之后单独验证自身rollout；保留动作与结构，才扩124f和10/20s。
- **效率：** 同硬件/offload、warmup后至少3次；含prefix/history重算、commit、传输和VAE的整视频/首块/每块延迟，分别报告forwards、显存与CPU存储。

最多3张项目GPU；旧队列与fixed/native-dual正控已退出；native-single局部正控也已结束，匹配5latent已结束，当前新12latent后续窗口使用GPU1/5；未启动新训练。[当前执行证据](submission/reports/stage1_anyflow/local_topology/README.md)。不因空闲卡增加候选或预算。原会议展示仍有效地呈现旧结果与限制，不提前改成功结论。

[完整可执行协议](submission/reports/stage1_anyflow/three_experiments/PROTOCOL.md) · [旧密度实验终态](submission/reports/stage1_anyflow/fm_density_control/FINAL_RESULTS.md) · [旧计划归档](submission/reports/stage1_anyflow/three_experiments/plan_archive.json)
