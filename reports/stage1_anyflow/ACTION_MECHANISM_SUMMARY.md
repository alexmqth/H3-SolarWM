# 当前chunk A/D反事实：对齐正确，动作差分方向仍不匹配

**2026-10-09 E1 归因限定：旧“Original”对照不都等于原生 H3 推理函数。** `field_factorization/field.py` 和 `real_abot_fm/geometry_primitives.py` 的 Original 分支也显式使用 `fixed_prefix_timesteps=True`，把 text/action 的 native time 固定为 1；发布 H3-World 默认让它们跟随 video time，即 `1−sigma`。所以“同权重只改 mask”的数值仍是有效的匹配对照，但只证明**在该原型时间条件下**依赖图变化的相对影响，不能将全部差异归因于原生 H3 的 attention 因果化。`generated_action_geometry128` 同时保存 matched/native 两类 teacher，不能据此否定该页全部结果。

SolarWM 的 Stage0.5 使用 video time，Stage1 使用 clean text time；这是需要训练适配的官方阶段策略，并非其实现错误。E1 正在分别校准时间条件、原生首帧条件与可见窗口。在完整 12-latent 当前窗口、其余条件相同的对照中，仅恢复原生时间使 A−D 从约 0.021764 增至 0.798524，但 A 仍约 −0.000594，方向未通过；也不证明后续 causal chunk 已恢复。见[本轮证据](local_topology/README.md)。下方旧数字保留，按各自协议解释。

**2026-10-09补充：真实FM密度对照的GT/generated各18点也已完成。** 主看后两个有历史chunk各12点，固定旧step00-generated状态的新shift2.22 FM48整体cosine为0.996377，当前A/D差分cosine仅0.035019、差分范数为teacher的0.672755（旧shift12为0.996389 / 0.029239 / 0.645296）。动作非零，但方向仍弱相关；小幅cosine变化没有建立可靠动作能力。同样12点GT历史delta cosine0.011453→−0.002418。首块无历史6点单列，完整18点也保留，不能混用分母。

新旧history/state/action pair/endpoint匹配、只读KV、A/D独立历史重建、重复输出检查通过。噪声54点扫描呈现低/中段微好、高段变差的取舍；改变sampling density没有修复动作geometry。旧收据缺完整teacher-output/anchor tensor hash、teacher重算历史而student固定KV、插值状态不是solver捕获、BF16小差分等限制仍保留。见[完整同状态结果与图](fm_density_control/GEOMETRY_RESULTS.md)。两条GT30与两条generated30已逐帧评审：前者未见明确改善，后者第二场景23帧后仍人物分解。其余8步/停车场评测仍在原队列中；完整A–D尚未完成。

以下为此前机制证据，模型/数据协议不同，不能将跨表数值串成学习曲线。

2026-10-08 23:30。本页直接回答“固定generated history和当前noisy latent，只换当前chunk A/D”的机制问题。**该验证已完成，支持优先调查因果化后的动作条件函数，而不是再泛称缺一次action alignment。尚未证明完整Stage1或Stage2效果通过。**

## 最直接的同状态结果

部署persistent-KV路径，AnyFlow128在`r=t`输出瞬时velocity，与Original H3 teacher的同语义输出对比。固定两条A/D generated history，在后两个chunk×3个sigma上共12点，只改当前chunk动作。

| 检查 | 已观察到的结果 | 含义 |
|---|---|---|
| 当前动作时间跨度 | latent5–9对应RGB[17,34)，latent10–11对应RGB[34,39) | 未发现本实验的chunk/action错位 |
| 真实attention mask | 当前video能直接读取当前action；action反馈能读取所属video | 动作条件有直接入口；当前设计与Original仍有差别 |
| 固定history KV | A/D分支内完整hash与commit计数不变；参数version不变 | 干预没有顺带修改历史缓存 |
| 重复和未来内容负对照 | 重复输出RMSE=0；只改future action内容时当前输出RMSE=0 | 检查固定layout下的重复性和内容泄漏 |
| 当前A/D实际影响 | student/teacher delta范数比0.7935–3.5563 | 动作有非零影响，不能称作完全漏传 |
| 差分方向 | cosine均值0.03822，范围−0.10234–0.20505 | 与Original动作差分整体弱相关 |

Original使用匹配dual anchors、prefix时间与全局位置。保留原native条件的另一组teacher平均cosine0.02968，结论相近但单点有差异。它们是同seed/场景下相关状态，不能当独立大样本统计。

原始检查同时保存了finite区间输出；上表**只比较`r=t`瞬时velocity**，没有将AnyFlow区间平均速度直接当Original瞬时速度。当前状态是保存generated endpoint和固定noise的插值，不是原solver实际中间状态。[完整检查与限制](generated_action_geometry128/INTERPRETATION.md)。

## 后续对照把问题定位到哪里

| 实验（各行协议不同，不能拼成学习曲线） | 整体velocity cosine | 当前A/D delta cosine |
|---|---:|---:|
| 同Original权重，只改causal依赖；完整输入范围受控的12状态 | 0.996253 | 0.060153 |
| 上述协议，匹配旧FM32 / AnyFlow32 | 0.992481 / 0.992425 | 0.054469 / 0.058434 |
| 真实ABot FM桥接，固定step00-generated 18状态，FM0→48 | 0.994733→0.995210 | 0.012540→0.009336 |
| 同一真实FM桥接，固定GT 18状态，FM0→48 | 0.988193→0.989033 | 0.017516→0.017641 |

第一行说明明显失配在AnyFlow更新前已经出现。第三、四行说明有限的真实视频FM适配改善整体拟合，却没有恢复动作差分；不是证明普通FM永远无法适配。第一组完整祖先重算不是部署KV等价性验证，两类证据单独列出。

此外，隔离的own-action＋公共prefix读取当前video候选，在无历史首块恢复Original输出identity，cosine=1；但后续12点仍仅0.008548（原路径0.028777）。不能把首块identity平均进去后宣称跨chunk修复。[跨chunk候选](current_prefix_candidate/INTERPRETATION.md)没有生成视频，也没有进入默认实现。

实际视频与机制结论一致：真实FM48停车场30/8 steps/chunk的A−D分离度为−0.018685/0.017456，Original参考为2.023377；30步人物大体保留但A/D近同向，8步后段仍重影。[全部视频和验收](real_abot_fm/FM48_COMPLETE_REVIEW.md)保留失败结果；Original视频有旧精度/条件协议限制，FM0/48才是严格匹配的训练前后对照。

## 为什么不能只说“action表征权重错了”

Original双向teacher允许当前动作影响当前video，继而改变历史hidden states，再反馈当前输出。Causal student的过去raw KV已经clean commit，当前动作不能回写历史表示。即使输入raw history相同，这两者的内部条件函数也不同。公共prefix反馈与past-action可见性也发生变化。

所以当前结论是：**action alignment与缓存机械检查通过，但动作条件velocity geometry没有迁移好；失配包含因果依赖图变化及其适配不足，尚未定位为某一层权重唯一出错。** 小动作差分还受BF16数值影响；主对照已统一SDPA，原生Flex与SDPA差分cos约0.84–0.87，不能把重复误差0解释成不存在精度误差。

## 新增噪声证据与后续门槛

[54点固定噪声扫描](fm_noise_audit/INTERPRETATION.md)已完成。FM48 raw MSE在52/54点下降，但只下降约1.47%；低sigma的Original误差同样升高，纯噪声起点有更大的causal额外误差。因此不能仅靠低噪声upweight解释或修复动作方向问题。

后续B适配应固定architecture并解耦sampling density与loss weight，保留分sigma、同状态A/D及完整39帧三套证据，不再把total loss或整体cosine改善当通过。C需要可信局部causal field后再验证AnyFlow；D需要有效teacher/critic分布监督及正确生成Jacobian。Stage1不必提前消除全部长时漂移，但当前连局部动作门槛尚未通过，不能将失败全部归因于缺Stage2。

[A受控对照](field_factorization/RESULTS.md) · [真实FM48几何](real_abot_fm/FM48_GEOMETRY_RESULTS.md) · [A–D当前状态](ABCD_STATUS.md)
