# 固定text/action clean time下的局部正控：未通过

两份Original-generated reference、每份三chunk各fork A/D、30steps全部完成。新T2保留Original directed attention、每个sigma重算可见窗口、无未来块输入，但沿用了原型的text/action clean-time覆盖。**不能把它无保留地称为原始H3推理函数。**

| 固定历史来源 | 当前chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
| A | 0 | -0.907143 | -0.897766 | -0.009377 |
| A | 1 | +0.490194 | +0.478376 | +0.011818 |
| A | 2 | +1.656709 | +1.657297 | -0.000588 |
| D | 0 | -0.907143 | -0.897997 | -0.009146 |
| D | 1 | -0.963516 | -0.963740 | +0.000224 |
| D | 2 | -0.608937 | -0.609612 | +0.000675 |

后续chunk里，参考历史A时两种当前动作都向正方向，参考历史D时两种当前动作都向负方向；当前A/D差很小。首块也未恢复方向。完整静态图中人物与停车场大体可辨，没有此前generated8的严重半透明分解；但动作门槛失败。39f串接在17/34帧恢复reference历史，有可见状态跳变，绝不能以它证明自由生成连续性。

扩大当前已知动作窗口到12latent/39RGB后，A=-0.745426、D=-0.767189，差约0.021764，仍同向。两条完整39f静态图中人物存在，动作几乎相同；窗口长度单独不能解释这次失效。它的控制粒度是39RGB，不是5latent在线原型。

源码追踪发现：发布H3-World的model_fn默认text/action time跟随video time(1−sigma)；原型fixed_prefix_timesteps=True将全部text设1。SolarWM Stage1确实采用clean text time，但其Stage0.5采用video time，需要经过训练适配。当前受控输入差异与attention mask是独立因素。新增5项CPU检查确认本30步网格上切换此flag只改变text/action时间，anchor/audio/history/video行时间不变；clean commit sigma0时anchor1 vs .999差异另记。

下一步已启动同权重/同anchor/同窗口的native text/action time对照，不先改anchor或加训练。原固定时间版本的geometry runner被正控门槛挡住，没有启动。旧几何结论仍是“在已固定的原型时间条件下改mask的相对影响”，不能直接据此声称完整复刻了原始H3动作函数。

[固定历史A局部视频](positive_A/AD_local_forks.mp4) · [固定历史D局部视频](positive_D/AD_local_forks.mp4) · [12latent窗口校准](window12_calibration/AD_window12.mp4)。全部27个MP4完整解码通过，原始指标与视频哈希保留。不是实时播放评审，没有新训练、没有Stage1验收通过。
