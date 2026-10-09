# 136有限预算对照已完成：未通过视频/动作验收

[完整结果与时间/显存/动作表](FINAL_RESULTS.md) · [8步四列A/D视频](original128_control_auxiliary136_8step_AD.mp4) · [4步四列A/D视频](original128_control_auxiliary136_4step_AD.mp4)

原loss136的8/4步A−D为0.580959/0.196324；一致性辅助136为0.635174/0.199916。4步A仍负，两组在18–38帧重影/雾化未解决；全39帧及对应帧静态复核，非实时播放。全部对比视频H264/yuv420p/24fps/faststart完整解码通过。最新136不能替换正式会议demo。

两组从同128权重/Adam/RNG各新增8更新，保存132/136。固定39帧、5+5+2 latent chunks、clean-history训练、generated-history推理、RGB dual、CPU raw KV、8/4步、seed13。全52层rank8 QKVO/FFN bank训练，visual/action/target-time冻结；不改architecture、anchor和采样。

候选只新增 `0.25*MSE(v_finite(z,.240780899,0), frozen128_diagonal16_average_velocity)`。这是官方AnyFlow之外的实验性辅助，不是Stage2。diagonal不是GT，固定参考fit不等于当前模型self-consistency，等更新数不等算力。

[两组训练报告](TRAIN_FORKS_RESULTS.md)保留128→136逐更新、真实恢复、随机流及冻结参数检查。逻辑CPU RNG/时间对核验通过，但实际GPU噪声未直接记录hash。CPU零权重等价、随机小H3六步history反传等测试不冒充视频效果。

[双局部指标](../dual_metric136/RESULTS.md)：低噪声当前自一致性18.12%→17.90%，finite伪GT距离0.058802→0.058319，只有小幅改善；没有转化为视频修复。[噪声覆盖检查](../dual_metric136/NOISE_COVERAGE.md)已完成。按用户最新指示，下一项优先固定generated history/noisy state的当前chunk A/D差分与Original teacher比较，排查时间对齐、直接attention和KV，暂无新训练。

源目录：`H3-World/outputs/2026-10-08-15/stage1_interval_consistency_candidate/`。归档不含大权重、完整runtime或target tensors；脚本执行依赖原目录。项目最多3张GPU，训练预算已止于136，Stage2未启动。原始运行JSON的NOT_REVIEWED是执行器的占位状态；实际人工静态验收见visual_review.json/final_results.json。
