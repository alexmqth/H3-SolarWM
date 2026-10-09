> 2026-10-08 16:04：本诊断及8/16细分检查全部完成，原进程均已退出；以[完整结果](RESULTS.md)为准。原始低噪声差异在细分后仍存在；参考不是GT，有限map到教师伪标签反而更近。随后启动[有限预算一致性对照](../interval_consistency_candidate/README.md)，尚无新视频结果。以下保留15:32启动协议。

# AnyFlow128有限区间只读诊断

2026-10-08 15:32：GPU5/6重新确认空闲后，分别启动A/D，PID173171/173176，状态running。没有新增optimizer更新、没有Stage2训练、没有新视频结论。启动时1001项冻结输入验证通过；CPU解析场与随机小H3检查通过，不等于真实33B验证完成。

源目录：`H3-World/outputs/2026-10-08-15/stage1_finite_interval_probe128/`。
依赖与checkpoint沿用`stage1_parallel_resume68_to128/training_runtime`及`train_128/step_128`。本目录的脚本是归档副本，实际执行入口在源目录，因为脚本用相对位置定位workspace。

协议：

- 每动作三个chunk，5+5+2 latent；全部历史与RGB dual anchor来自Original30 teacher。
- CPU raw KV、causal action rows、feedback ON、同一128权重与h3_fp32。
- 固定teacher latent和已保存初始噪声，以`z=(1-sigma)*clean+sigma*noise`构造输入。这是clean-history训练支持上的局部诊断，不是完整自生成轨迹。
- 从native8、shift2.22网格取第0、4、7个区间，覆盖高、中、低噪声。
- 每区间从完全同一状态出发：1次正常finite map，与4/8等分的r=t Euler积分。
- 每动作9个case、117 noisy forwards + 3 clean commits；不做训练、不改采样器或正式视频。
- 实际初始状态/锚帧/噪声/prompt/audio/action有hash；每个chunk检查KV内容hash与commit次数不变；最后检查模型parameter version不变。

输出`probe_A.json`、`probe_D.json`包含endpoint/velocity RMSE、速度方向cosine、4→8参考变化、运行时间与显存。引用任何偏差前必须同时看4/8参考差异。参考是**同一个模型的数值自洽性**，没有独立GT；若细分结果本身未收敛，不能据此判定finite-map错误。不能由内部残差下降推断视频改善。

CPU验证：常速度场精确一致；已知解析有限映射的时变场展示Euler离散误差随细分减半，避免把数值积分误差误判成模型误差；故意给finite map加偏置时能检出差异，diagonal参考保持相同；随机小H3三个chunk/三个区间的KV和模型参数不变。收据`cpu_self_test.json`、`cpu_h3_integration.json`。

后续：先读完A/D有限区间结果，必要时只加数值参考的收敛检查；有系统失配证据后才决定小预算Stage1一致性辅助实验。仍须用正常finite-map 4/8步视频评估画质/动作。若局部能力合理而free-running历史仍持续漂移，则进入当前AnyFlow初始化上的Stage2对照，无需等待Stage1消除全部长时漂移。
