# FM48 generated-history 30步：两个自然场景

2026-10-08 22:41。完整39帧已完成，静态检查所有帧及GT/Original/FM0/FM48对应0/8/16/24/30/38帧；不是实时播放评审。

第一场景（118e…A_1140）：FM48人物与场景总体保持，建筑/树木仍有漂移和形变；FM0也已能保留人物。边界RGB MAD17.0996→9.5143，整体灰度MAD9.8453→7.7584；运动有所减少，不将帧差下降单独当作视频质量改善。平均水平flow0.4666→−0.1425，原始参考2.2888；这是联合动作自然片段，不套纯A/D方向gate。

第二场景（dfec…D_1750）：FM48约23帧起人物上半身出现红色破碎/残影，26–35帧明显分解，37–38帧人物结构近乎消失。FM0也有同类失败，没有修好。边界RGB MAD19.8830→14.1391反而下降，不能靠这个数字判PASS。灰度MAD13.0319→11.6552，flow−0.7597→−0.7472，Original−6.7283。

两条都用相同noise/actions/initial condition作训练前后对照：39RGB、12latents、5+5+2chunk、30/chunk、generated history、RGB dual、CPU raw KV、h3_fp32。90 noisy forwards+3 clean commits、CPU KV6484.13MiB。第二条FM48耗时1010.5s after-load，明显高于baseline565.9s；共享GPU期间其他作业增加，单次墙钟数不可直接用于训练效果/架构加速归因。

[完整四列视频及指标](report/trained_complete_generated30/README.md)

这两条已足以否定“FM48解决generated-history画面崩溃”的说法。8步结果仍待收齐；不以未完成结果作为证据。
