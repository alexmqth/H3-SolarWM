# Next Plan v4：先恢复局部动作信息流

当前顺序固定为：**可信 Causal H3 → AnyFlow few-step → on-policy Stage2**。第一项要求在正确历史和充分采样步数下，多个窗口的动作响应与人物结构成立；不要求先实现20秒稳定自由生成。

2026-10-09 16:16：E2两臂各4更新、六组局部视频评测及48次held-out诊断全部完成。停车场两份历史的A/D符号均保留，但A分支重影仍在，FM+action没有一致优于FM-only；局部动作＋结构联合gate仍为No-Go。本轮不自动扩训，不进入AnyFlow/Stage2。

## 三个有决策意义的实验

| 实验 | 唯一研究问题 | 控制与验收 | 当前状态 |
|---|---|---|---|
| E1 局部信息流 | Original H3的信息流怎样在不见未来时保留下来？ | 比较局部双向/跨窗因果拓扑；删除未来视频和动作后再refine；检查首窗与后续同状态A/D，不只检查首窗 | T1/T2因果和执行检查成立。T2/N后续窗方向改善，但人物重影使联合gate未过 |
| E2 动作后果监督 | 正确拓扑上，显式动作监督是否比普通FM更有用？ | 同初始化、数据、噪声、参数bank和预算，仅loss不同；观察真实transition，不相减不同history的teacher输出 | FM-only / FM+action各4更新完成；两份停车场history与两份held-out GT-history评测全部完成；未显示额外动作损失收益 |
| E3 AnyFlow→Stage2 | 局部能力成立后如何减少采样、适应自身history？ | 先可信30-step对照4/8-step；再比较GT/reference history与完整generated history，最后测端到端效率 | 前置联合gate未过，暂不投入GPU预算 |

## E1已经得到的证据与约束

[完整历史条件结果](reports/stage1_anyflow/history_conditioning/VIDEO_RESULTS.md)。T2保留当前可见窗口内的Original有向attention关系，逐sigma重算可见prefix/video hidden；只使用已观察历史、当前动作和当前噪声，不使用未来状态/动作，不回改已显示RGB。

在Original权重、12latent当前窗、single I0、30步下，N把已知历史临时变为`(1−sigma)H + sigma*epsilon_H`，匹配原生video时间。两份停车场reference history的后续窗口A/D方向均正确，但A分支仍有人物重影。它证明历史条件显著影响动作响应，不能证明局部画质或完整E1通过。Reference history来自Original生成，不能写成真实GT或student自由rollout。

T2/N **不复用persistent hidden KV**，CPU hidden KV为0；必须把重算成本纳入效率评测。持久KV方案保留为历史路线，不作为本轮已获得的加速收益。

## E2冻结协议及当前预算

[实验文档](reports/stage1_anyflow/real_transition_windows/README.md)，工作目录`H3-World/outputs/2026-10-09-06/stage1_real_transition_windows/`。

- 真实6train+2validation，episode隔离，同episode完整124RGB不重叠；保留联合WASD/相机条件。12latent窗口对应RGB `[0,39)`、`[39,81)`、`[81,120)`，不把120帧写成124帧rollout。
- 24个真实VAE前缀差0、6个未来RGB反转差0、24个未来文本内容/长度检查通过；17项CPU梯度/因果/hinge等价检查通过。窗口内派生camera F是观察后果代理，不是独立用户速度指令。
- Original H3＋released action LoRA，只更新tail8 QKV/out共10,092,544参数；普通FM、T2/N历史、固定noise、FP32 master、LR2e-5、logical batch2，两臂均4更新。全部梯度replay误差0，初始bank完全相同。
- 定义`e(a)=MSE(v(a),epsilon−x0)`；只交换当前A/D，保留history、noisy state、W/S及camera。动作项`lambda*relu(margin+e(a)−e(a'))`是观察后果的排序监督，**没有**另一个动作的真实视频。
- margin=`9.3068927526474e-5`，lambda=`2.4858908311120174`，只用train校准并在训练前冻结。Held-out不用于调系数。
- 原预算上限16/臂，但4-update实际生成未通过前不延长。没有增加模块、窗口大小或采样步数。

## 已完成的评测与下一决策

1. 停车场A-history、D-history各固定同一H/z，fork当前A/D；三列对照Original权重下的局部N、FM-only4、FM+action4。每条30steps，完整看42当前RGB及原尺寸人物细节，不以flow变大代替结构改善。
2. 两份真实held-out GT-history24，各生成12latent/39RGB，保持实际联合动作；看边界、人物和场景。联合相机动作不能套纯A/D的flow符号。
3. 两验证状态×四sigma×正/负动作×三参数bank，共48个诊断forward已齐；同时报告正确动作绝对FM、错误动作FM、gap与action-delta RMS，检查是否仅抬高负例误差。当前两臂正确动作误差仅小幅下降，动作项无更好排序证据。
4. 两臂同协议结果已齐：动作项无一致收益，A分支重影未解决。本轮停在4更新，不宣称E2成功，不自动扩训。8个训练microbatch无纯A/D、3个含相机；后续先审可靠动作后果与时延，不能靠调验证集系数或增加更新绕过监督缺口。

六组视频及48次诊断均完成；原训练/评测进程与两队列已核实退出。全程最多3张项目GPU。新实验前重新检查资源，不使用旧PID收据启动重复作业。

下一实际步骤：先复核train-only纯动作候选的动作时序和可见后果，并补齐可靠同状态参考的准入条件，再冻结一次E2修订。CPU已找到A/D各一段候选，但尚未通过训练准入；不新增网络、anchor、solver扫描，不进入E3。参见[本轮最终报告](reports/stage1_anyflow/real_transition_windows/FINAL_RESULTS.md)。

## Go / No-Go

| 验收点 | 通过条件 |
|---|---|
| 局部动作 | 多个chunk/同状态动作干预一致改善，实际生成方向正确；首窗或一组flow符号不足 |
| 局部画面 | GT/reference history＋充分采样下人物结构基本成立，无严重分解；分别声明历史来源 |
| Few-step | 4/8-step接近已可信的causal30-step，兼看动作与结构，不只看finite/diagonal residual |
| Generated history | 不依赖GT重置的完整自由生成明显优于旧基线 |
| 效率 | 同硬件/offload，warmup后多次整视频、首块、每块延迟，含重算/传输/commit/VAE；不按30/8算加速 |

会议主demo保持原验收状态；本轮局部诊断归档到reports，不作为124f最终结果。当前完整Stage1及研究目标仍未完成。
