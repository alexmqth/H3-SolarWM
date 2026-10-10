# EXP-009最终Judge审核：FM4有限续写可用，AF4当前配置不验收

2026-10-11 05:00 HKT。任务已完成，**普通FM4 AA/AD73有限续写可行性PASS、quality PARTIAL；AF4 AA为PARTIAL，但AD C3持续人物分解，生成FAIL，联合可行性不验收。** 当前AF4配置停止，不追加2NFE、checkpoint扫描或训练。已有AF8有限可行性结论保留，不把4NFE失败推广为AnyFlow整体失败。

## 实际协议与独立证据

两个模型都使用EXP-006 FM8的同一首12 clean latents与前39 RGB，各按自身权重建立C1 raw KV。FM4为Original H3 + released Action LoRA，冻结interval_cached与native scheduler；AF4加载EXP-007 AF2 step32配对QKV/target-time，用next-sigma finite map。Single I0/native time/current-prefix/own-action/action feedback、Global、strict causal、persistent KV不变。

4-step sigma为1、.8694517212、.6894410400、.4252873230、0。C2同clean历史/动作/初噪声比较，C3各用自己生成的C2。属于两个方法的有限续写比较，不能称全程FM4/AF4首窗生成、训练目标单因素消融或跨场景泛化。

[最终CPU审计](c3_audit.json)通过28项来源、checkpoint配对、原始输出摘要、所有50层cache indices、各模型自身KV来源、同seed13 noise/prompt、历史RGB逐值不变以及所有73帧24fps/单调PTS完整解码检查。全部GPU进程已结束，无失败重试。工程PASS与下面生成结论分开。

## 全帧视觉与动作结论

Judge检查8段共136张新增帧、边界与必要原分辨率末帧；见[逐段记录](visual_notes.json)。首39帧由共同FM8提供，不重复计为本轮新生成证据。

| 配置 | C2 AA / AD水平flow | C3 AA / AD水平flow | 观察与结论 |
| --- | --- | --- | --- |
| 普通FM4 | +.706 / −.190 | +.336 / −.618 | A/D方向可辨，D C2较弱；人物/停车场基本可用，腿部透明残影、姿态/透视与边界跳变保留。有限可行PASS、quality PARTIAL。 |
| AF4 step32 | +.489 / +.162 | +.268 / −.0075 | AA仍可辨但C3人体持续透明；AD C2方向弱/含混、场景纹理扭曲，C3人物持续分解成大片透明碎片、控制近乎消失。AD C3 FAIL，整体无优于FM4的证据。 |

AF4结论来自持续结构缺失和完整画面观察，不仅因为一个flow反号。AD C3末帧[原分辨率](af4_AD_C3_last.png)显示人物结构被大量重叠碎片破坏，明显超出普通肢体细节缺陷；背景停车场仍部分可辨，不称全帧噪声。FM4普通缺陷按用户当前可行性尺度接受，不追加调参。

## 实际成本与边界

共32sampling+6clean commit=**38forward、8VAE、0update，782.290028秒=.217302786GPU小时**，低于.6GPUh；allocated峰值26.33646GiB。C1 CPU video KV 6,799,104,000 bytes，C2 commit后9,632,064,000 bytes；当前3块未触发淘汰，不借此宣称滑窗能力。

每块sampling约22.5–24.9秒，decode约4.9–6.8秒；完整阶段还包含模型重复加载、cache I/O、保存等。与已有8步的采样时间仅作范围说明，不将调用数减半宣称为公平E2E加速。详细逐块计时、边界MAD、显存/KV见审计JSON。

## 决策与下一步

此轮充分回答了固定短训student在4NFE是否值得继续的问题：没有出现支持继续投入的联合收益，停止当前AF4步数探索。保留普通FM4短续写对照，正式V3 Baseline、SW-G30、全程FM8、AF8证据冻结。

下一EXP-010回到主线，准备全程普通FM8与SW-G的124/158帧联合验证，复用EXP-006 AA73并按固定历史延长；零新训练，独立编号与预算，GPU分阶段放行。这检查已可用低步数方法与真实KV淘汰的组合，不挽救已失败的DMD或AF4配置。
