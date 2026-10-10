# Judge验收：EXP-006/v1 全程普通FM8

2026-10-11。**accept：从新生成首窗开始的AA/AD73帧有限可行性通过，画质/严格连续性PARTIAL。** 任务完成，0训练，正式V3 Baseline及SW-G结果保持冻结。下一个执行任务为独立V3-AF初始化与单步训练，不继续扫FM步数。

## 本轮回答的问题

EXP-004的首39帧来自30步，只证明8步续写。本轮首12 latent也从保存的原始噪声用8步生成，再用自己的sigma0 clean KV续AA/AD两块至73帧。全程Original H3 + released Action LoRA，Single I0、native timestep、own-action/action feedback/current-prefix、Global位置与strict causal/persistent raw KV不变。

## 独立证据

Judge查看了完整首39帧与四段新增17帧、各边界及原分辨率末帧，核对全部五段输出的RGB hash、MP4帧数/24fps/递增PTS。后续旧RGB逐值不变；C2读取新首窗cache，AA/AD的C2共享同cache/尾部噪声，C3各接自己的C2；全50层indices与cache容量检查通过。冻结参数未改，源码/config/manifest与实际运行记录一致。

| 路径 | C2水平flow | C3水平flow | 视觉观察 |
| --- | ---: | ---: | --- |
| AA | +0.8458 | +0.5052 | 主体/停车场可辨；C2开始姿态和透视跳变后恢复，C3有持续肢体透明残影 |
| AD | −0.3206 | −0.8265 | D方向和切换响应可辨，C2幅度较弱；切换及C3边界跳变，主体可用 |

光流只辅助方向判断，不作精确控制率。没有持续主体消失或整体场景崩溃；普通质量缺陷按可行性尺度保留，不追加调参。只覆盖单停车场seed13、73帧约3.04秒；不把它延伸为124/158帧全程FM8或成熟控制能力。

[机器审计](completed_audit.json) · [逐帧记录](visual_notes.json) · [首窗审核](first_review.json) · [最终机器判定](final_review.json)

## 成本

实际40sampling+3clean commit=**43forward**，**5VAE**，**492.856秒=0.136905GPU小时**，低于0.75GPU小时；单GPU0顺序执行，无重试/训练。GPU allocated峰值**26.061GiB**。首窗历史KV为6,799,104,000 bytes，第三块读取C1+C2约9.632GB；本任务未触发窗口淘汰，滑窗证据在EXP-005。

| 块 | sampling秒 | commit秒 | decode秒 |
| --- | ---: | ---: | ---: |
| 共享C1 | 47.21 | 8.52 | 3.50 |
| AA C2 | 39.83 | 5.83 | 5.25 |
| AD C2 | 38.98 | 5.67 | 5.06 |
| AA C3 | 50.05 | 0 | 6.63 |
| AD C3 | 41.76 | 0 | 6.42 |

全部内存/耗时精确值见机器审计。[首屏与计时边界](latency_scope.json)：首39帧MP4最后写入相对阶段开始约91.39秒，只是文件可用时间代理；没有测交互首帧显示。各链条阶段占用相加AA约298.79秒、AD约289.70秒，含重复加载与缓存I/O，不含人工检查间隔或另一分支，不能称连续单进程公平E2E加速。

## 视频与比较范围

- [全程FM8 AA73](../artifacts/videos/AA_FM8_73.mp4)
- [全程FM8 AD73](../artifacts/videos/AD_FM8_73.mp4)
- [保存的30步参考 / 新首窗FM8：AA](../artifacts/videos/AA_FM30_saved_first_vs_FM8_new_first_73.mp4)
- [保存的30步参考 / 新首窗FM8：AD](../artifacts/videos/AD_FM30_saved_first_vs_FM8_new_first_73.mp4)

比较共享原输入/噪声/动作，但首窗及后续生成历史不同，按闭环方案对比，不称同raw KV单状态消融。旧30步参考不重新生成；本轮不作速度排名。

## 下一步

普通FM8已成为低成本有限对照。EXP-007先建立新的target-time-conditioned student，在原V3协议上做初始化对角回归及一个finite-map训练更新；后续是否继续32updates、是否进入DMD，由真实训练成本和视频结果决定。用户已授权夜间持续推进至09:00，无需重复征询授权；每个阶段仍按明确预算执行。

运行原始JSON保留`complete_pending_visual_review`保证SHA不变，正式接受状态以本审核为准。
