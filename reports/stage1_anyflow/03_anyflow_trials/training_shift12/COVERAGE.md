# 训练时间区间覆盖：原协议2.22与官方Stage1的12

本表通过CPU按实际step00的logical RNG状态重放预定16/32/64次采样，并按真实每chunk的latent尺寸消耗video-noise RNG。审计时已完成的36个sample sigma/r与真实训练逐项相同；后续行是固定协议下的预定采样覆盖，不是已经完成的训练。

| Updates | Action | 类型 | 样本数 | shift2.22 中 sigma≥0.9 | shift12 中 sigma≥0.9 |
|---:|---|---|---:|---:|---:|
| 16 | A | endpoint | 8 | 0 | 2 |
| 16 | A | diffusion | 16 | 2 | 12 |
| 16 | A | flow_map | 8 | 1 | 5 |
| 16 | D | endpoint | 8 | 1 | 3 |
| 16 | D | diffusion | 16 | 4 | 12 |
| 16 | D | flow_map | 8 | 2 | 3 |
| 32 | A | endpoint | 16 | 1 | 5 |
| 32 | D | endpoint | 16 | 1 | 8 |
| 64 | A | endpoint | 32 | 3 | 13 |
| 64 | D | endpoint | 32 | 5 | 20 |

A在16次更新里的endpoint高噪声样本为0，但diffusion仍有2、general-map仍有1；不能写成模型完全没有高噪声监督。阈值0.9只是描述覆盖，不能当作新的验收指标。

官方配置为SolarWM/configs/examples/minimax_h3/stage1-158f-lora384-w6-sp2.yaml：video_timestep_shift=12、global_batch128、rank384；本项目仍是logical batch4、rank8、两条39f伪真值、16次小实验。这里只采用官方的训练time shift，未复现其完整recipe。

由此准备的受控候选将训练pair采样与对应Gaussian权重改为12，公共validation和实际inference继续2.22。架构/初始化/数据噪声/optimizer/更新数都不变；不将训练分布覆盖差先验宣布为视频失败原因。

工程证据：17项相关CPU测试；默认入口与旧full-history4逐bit一致，shift12连续4次vsstep2恢复到4次的权重/Adam/RNG/loss/gradient逐bit一致；改变training shift被exact-resume入口拒绝。实际GPU初始四套adapter、三种RNG、公共验证sample已匹配，见gpu_initial_audit_early.json。

GPU1 fresh16正在运行，尚无该候选视频，验收仍使用A/D方向、分离度及完整人物/场景。
