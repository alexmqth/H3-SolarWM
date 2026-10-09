# FM密度对照：六条自然场景视频全部完成

2026-10-09。两个held-out ABot场景×GT-history30/generated-history30/generated-history8，共六条新候选视频，各39RGB帧、24fps。新旧FM48使用相同初始化、rank8 bank、真实视频训练课程、固定weight函数，只改变训练sigma采样shift12→2.22。**这项改变没有建立通过验收的causal模型。**

| 历史与步数 | 植被/建筑场景 | 石墙/街道场景 | 对结论的作用 |
|---|---|---|---|
| GT-history，30/chunk | 人物可辨，背景与运动偏离，17/34边界重置 | 人物可辨，但视角/运动与Original不同，边界重置 | 新旧无明确视觉质变；GT oracle不能证明free rollout稳定 |
| Generated-history，30/chunk | 人物保持但运动偏弱 | 约23帧红色残影，24–36人体分解，37–38接近消失 | 足以否定本轮视频稳定性PASS |
| Generated-history，8/chunk | 人物可辨，背景横向拖影/重复轮廓持续 | 更明显模糊，23帧后人体与红色残影混合，末段结构不可靠 | 较少steps没有通过；30步也不足以根治 |

所有判断来自全部39帧静态接触表、对应帧和18–38放大帧；未作实时播放流畅度结论。MAD是运动/帧差，不能代替画质；自然片段含联合键盘/camera，不能用两段视频的flow差代表A/D正确性。

## 完整视频与逐帧证据

- [GT30：两个完整四列视频、指标和全部帧](report/natural_gt_30/README.md)；[人工评审](report/natural_gt_30/MANUAL_REVIEW.md)。
- [Generated30：两个完整四列视频、指标和全部帧](report/natural_generated_30/README.md)；[人工评审](report/natural_generated_30/MANUAL_REVIEW.md)。
- [Generated8：两个完整四列视频、指标和全部帧](report/natural_generated_8/README.md)；[人工评审](report/natural_generated_8/MANUAL_REVIEW.md)。

时间为单次共享主机且conditioning已缓存；allocated显存未拆权重/激活，CPU KV单独记录。8/chunk为24 noisy＋3 commits，30/chunk为90 noisy＋3 commits；Original30是30 full-sequence forwards，不能把步数简单相除作speedup。

[同状态动作几何](GEOMETRY_RESULTS.md)也未恢复：有generated history的12状态，新模型整体cos0.996377但A/D delta cos0.035019；GT有历史delta cos−0.002418。结合[54点noise取舍](report/noise/INTERPRETATION.md)，没有证据继续扩大该密度因素的48-update预算。

停车场纯A/D30/8尚在原GPU队列中，待完整收齐后补最终动作表。本页仅结案六条自然场景视频，不声称整个B评测、更不声称A–D目标完成。meeting保持现有版本，未推送。
