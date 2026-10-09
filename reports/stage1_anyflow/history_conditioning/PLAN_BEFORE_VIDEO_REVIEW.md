# Next Plan v4：先恢复局部动作信息流

2026-10-09 06:14：E1历史条件C/N的56次真实H3同状态探针全部完成；首窗identity、旧C重放、重复误差均0，CPU独立重算一致。干预改变了动作差分，尚不能判方向改善。GPU1/5正在窗口1两history×A/D的30步视频，预算120采样前向、零训练；窗口2仍受人工门槛约束。[当前执行](submission/reports/stage1_anyflow/history_conditioning/PROBE_RESULTS.md)。

2026-10-09 05:59更新。执行顺序按用户要求保持：**可信causal H3 → AnyFlow少步 → on-policy Stage2**。不继续旧FM48/AnyFlow136扩训，不要求Stage1先解决20秒漂移；但局部动作和人物结构必须先成立。

## 当前决策

E1的12latent窗口对照已完整结束，仍No-Go：首窗方向恢复，接入历史后部分窗口动作近同、部分分支肢体重影。不能继续增加窗口宽度，也不能拿首窗通过替代多个chunk验证。[全部结果](submission/reports/stage1_anyflow/coarse_window12/FINAL_RESULTS.md)。

| 实验 | 研究问题 | 通过条件 | 当前状态 |
|---|---|---|---|
| E1 局部信息流 | Original权重能否在不看未来的条件下保住当前动作 | 多个chunk同状态干预和实际方向可信；GT history+30步结构成立 | 机械路径与条件校准完成；多窗口效果未过，继续定位历史条件 |
| E2 动作后果监督 | 整体FM之外的可靠配对动作监督是否有帮助 | 同起始状态的真实/合法局部后果；正确动作拟合及held-out生成一起改善 | 设计保留，缺可靠局部配对参照，未启动 |
| E3 AnyFlow→Stage2 | 先少步，再适应自身生成历史 | 4/8接近可信causal30；free rollout优于自身对照；实测整视频成本 | 前置门槛未过，未启动 |

## E1 下一项：只改变历史条件协议

[完整有限协议](submission/reports/stage1_anyflow/coarse_window12/NEXT_HISTORY_CONTROL.md)，原计划登记时尚未实现；当前已完成实现、CPU与真实探针，视频运行中。保留本轮No-Go的冻结源码、视频与收据，建立新实验目录。

1. 现有C：clean历史H，历史video模型时间1，全部action/text时间1−sigma。
2. 对照N：临时历史输入`(1−sigma)H + sigma*epsilon_H`，历史video与action均用1−sigma。epsilon_H来自同fixture已知前缀的固定noise；A/D共用，不每步重抽。只更新当前chunk，保存历史与已发出的RGB不变。
3. 其他条件全部固定：Original+released LoRA、12latent、30steps/shift2.22、native prefix、单I0固定位置、同audio、布局、action路由。该干预改变历史的噪声和对应时间这一整套协议，不能声称单独找到time bug。
4. 先CPU验证无历史/零sigma一致、实际逐行时间、history只读、未来隔离。再在保存的同一A-solver state上fork A/D，两个后续窗口×两history×三sigma×两协议×两动作=48次，加identity/repeat最多8，诊断上限56次。
5. 探针变化不等于正确性。执行语义通过后，先窗口1两history共120次采样前向；方向和人物结构都改善才追加窗口2最多120。没有改善即停止这一因素，不扫噪声强度、不扩网络/窗口。高sigma下忽略历史、从I0重置，即使A/D符号恢复也FAIL。
6. 两后续窗口成立后，才做真实GT-history、held-out seed29与动作切换的局部验收；不能直接跳E3。

[CPU实际历史条件审计](submission/reports/stage1_anyflow/coarse_window12/history_contract_audit.json)确认了现有time处理，但它沿用pipeline原生retake语义，不是已证实bug。VAE两样本四边界共8项future-RGB检查差均0，当前不支持历史编码泄漏解释失败。两者都不是新视频改善证据。

## E2 不用错误teacher强行监督

配对必须来自同一起始状态、同history/noise/prompt，只换当前动作，且后果方向/画面已可靠。双向全长Original能看到未来，不能当无条件逐点差分真值；ABot不同轨迹不能直接相减。保留FM-only与FM+action-swapped transition ranking的有限对照设计，正确动作绝对误差、错误动作loss与实际生成共同检查，防止只恶化负例骗margin。[完整三实验协议](submission/reports/stage1_anyflow/three_experiments/PROTOCOL.md)。

## Go / No-Go与资源

- 局部动作：首块和两个后续块同时有可信响应，未来动作/video不可达，不能仅以flow符号或delta范数通过。
- 局部画面：GT history+30步人物/场景基本正常。Original-generated历史需单列，oracle边界重置不是自由生成。
- Few-step：可信causal30先成立，4/8的画面与动作接近它；finite residual下降不替代效果。
- Generated history：之后评估自身历史的退化，再决定on-policy Stage2及124f/10s/20s。
- 效率：同硬件/offload、warmup后至少3次；含条件构造、重算、commit、传输、VAE的整视频/首块/每块延迟。不能按step数比加速；T2无persistent hidden KV，必须披露重算成本。

最多3张项目GPU，不碰他人进程。旧coarse任务全部退出；新历史条件视频正在GPU1/5运行，没有新optimizer。会议演示保持既有结果与限制，未推送。完整研究目标与Stage1仍未完成。
