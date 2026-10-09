# 完整generated-history 30/chunk评审：密度分支仍有视觉失败

2026-10-09。两个自然场景均生成完整39帧。新旧全部39帧接触表和新候选放大18–38帧已检查；第二场景另查看旧候选相同放大帧。四列视频包含真实GT / Original30 / 旧shift12 FM48 / 新shift2.22 FM48，相同初始图、prompt、自然联合动作、video/audio noise。静态逐帧评审不声称实时播放流畅度结论。

**本轮密度改动没有修复free-rollout视觉失败，不满足人物结构保持的验收条件。** 不能把部分sigma的FM loss改善当成画质改善。

- 第一场景（118eb…A_1140）：人物在39帧内保持可辨，人物动作/场景位移偏弱，背景结构与植被仍变化；与旧FM48相近，没有明确视觉质变。逐帧图见[单场景评审](../../review/first_generated30/README.md)。
- 第二场景（dfec…D_1750）：约23帧开始红色残影，24–27帧身体逐渐被云状/颗粒状结构替代，28–36帧大部分主体轮廓消失，37–38帧人物接近不可辨。旧shift12同样在这个时间段明显分解，新shift2.22未修复。此处使用自己的generated history，与GT oracle边界重置是不同现象。

| 场景/方法 | Frame gray MAD | Boundary RGB MAD | Mean horizontal flow |
|---|---:|---:|---:|
| 植被 Original30 | 19.052133 | 18.753836 | +2.288792 |
| 植被 旧FM48 generated30 | 7.758413 | 9.514335 | −0.142475 |
| 植被 新FM48 generated30 | 6.940786 | 7.701437 | −0.217875 |
| 石墙 Original30 | 14.836259 | 13.811858 | −6.728256 |
| 石墙 旧FM48 generated30 | 11.655191 | 14.139110 | −0.747189 |
| 石墙 新FM48 generated30 | 11.560387 | 13.728716 | −0.839421 |

MAD下降不能解释为画质PASS；自然联合按键/camera的flow不等于纯A/D动作准确率。两条新视频分别载入后519.82s/502.75s，GPU allocated26002.32/26366.75MiB，CPU KV均6484.13MiB；各90 noisy＋3 clean commits。共享主机单次记录、条件预缓存；旧第二场景耗时1010.48s，负载不一致，不能把约2倍时差说成训练采样策略提速。

同状态机制已经另行完整检查：新FM48在generated后两chunk的A/D delta cosine0.035019，GT为−0.002418；变化幅度非零但未恢复方向。这与本组失败共同表明局部action适配与generated-history视觉稳定性都尚未建立。不能由这两条视频证明完整Stage2必然解决问题。

完整视频与指标见[报告入口](README.md)；第二场景放大图及源hash在[review目录](../../review/second_generated30/)。停车场纯A/D及generated8评测仍在原队列中；不增加本分支48-update预算，不替换meeting。
