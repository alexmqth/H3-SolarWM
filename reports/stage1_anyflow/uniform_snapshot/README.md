# Uniform sigma 对照：完成，未通过

只改变推理网格，保留旧AnyFlow trainable-time checkpoint、39帧、generated history、seed13、RGB dual及其他设置。Uniform4为[1,.75,.5,.25,0]，训练shift仍是2.22。

| Checkpoint | A flow | D flow | A−D |
|---|---:|---:|---:|
| step00 | -2.711084 | -2.140539 | -0.570545 |
| step16 | -2.695826 | -2.253384 | -0.442442 |

四条视频均完整解码39帧（1.625秒），各12 noisy forwards+3 clean commits；初始conditioning逐张量与对应原始H3一致。但抽帧均显示严重后段雾化及人物/车库结构丢失，动作与视觉gate失败。单独替换网格没有修复该pilot，也不能据此推断充分训练后的AnyFlow无效。

[全部四条的抽帧](completed_uniform_contact.jpg)、[带视频哈希的观察记录](visual_review.json)、[完整运行指标](run.json)。视觉观察使用frames0/10/20/30/38，属于明确的失败证据，未作播放质量PASS声明。单次共享GPU计时不用于公平加速比。

冻结时间MLP的16-update对照现已在GPU2启动。它是独立训练分支，尚无该分支33B视频质量结果。
