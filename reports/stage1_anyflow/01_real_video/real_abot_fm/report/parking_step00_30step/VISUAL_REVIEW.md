# 停车场30步：画面较完整，A/D分化仍失败

本结果来自新增adapter零更新的causal0，不是正在训练的FM48。39帧、3chunks、generated history、RGB dual、CPU KV、30steps/chunk。Original30是整段30步。

两条causal全部39帧contact sheet及Original/causal的0/8/16/24/30/38对应帧已静态检查：人物保持到末帧、车库主要结构在，没有中世纪场景的红雾分解；中后段人物缩小/朝深处移动，背景接缝和细节有漂移。A、D生成的视角轨迹与人物运动非常相似，未显示Original那样的左右分化。不是人工实时播放，不称为长视频稳定。

| 方法 | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| Original30 | +1.181253 | −0.842124 | 2.023377 |
| Causal0 30/chunk | −0.195516 | −0.186950 | −0.008565 |

因此，30步也未保留这组Original明确存在的action differentiation；视觉结构相对完整≠动作正确。A/D初始noise、audio、prompt、anchor、action rows与对应Original逐张量一致。Original旧legacy精度/固定首帧/联合prefix，causal为h3_fp32/RGBdual/cachedprefix，不能把整条pipeline差异只归因到一个mask或只归因到精度。后续causal0/48的匹配对照用于判断真实FM训练是否改善。

各causal为90 noisy+3clean、CPU KV6484.13MiB；A/D recorded wall925.19/856.24秒，GPU allocated16234.51MiB。单次共享主机、Original offload reserve未知，不能计算公平加速比或宣称显存节省。MAD作为活动/帧差记录，不充当画质分数。

[完整两行并排视频](original_causal_AD.mp4) · [逐条指标](metrics.csv) · [A对应帧](A_matched.jpg) · [D对应帧](D_matched.jpg)。8步baseline仍在运行；FM48训练后还会生成同设置回归，不提前报告训练有效或无效。
