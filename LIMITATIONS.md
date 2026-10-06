# 限制与诚实边界

1. 本项目不是 SolarWM 官方 Stage2 的完整复现。没有训练官方规模的 SGF/DMD、LoRA384、158-frame recipe，也没有声称达到 SolarWM 的质量或速度。
2. Stage2-lite 只是在共享 H3 backbone 上轮换 student、critic、teacher adapter 的 feasibility diagnostic。它证明训练链路可以运行，不证明 distribution matching 已经足够恢复动作。
3. 124 帧的视频证明的是 causal execution 和视觉稳定性。它们不能单独证明 W/S/A/D 的 image-space direction 正确。
4. Frame-to-frame MAD 和 signed optical flow 是运动/方向代理，不是 FVD、LPIPS 或用户研究意义上的完整视频质量评分。
5. action gate 目前没有通过：generated-history causal rollout 下 A/D score geometry 与原始 H3 teacher 不对齐。不能把 fixed-mix 旧 grid、RGB visual stable grid 或 Stage2-lite video 描述为“动作保真”。
6. 主要视觉 adapter 是在固定初始图像、prompt、seed/noise 和有限 action protocol 上得到的实验 adapter；它不是泛化训练后的完整 checkpoint。
7. 提交包省略了基础模型、数据集、缓存和 conditioning/latent 中间文件。拿到包后需要按 REPRODUCE.md 外部准备权重，不能只解压提交包就直接推理。
8. 更可信的后续方向是多 state、多 seed 的 action counterfactual supervision，或覆盖 student generated-history distribution 的正式 Stage2-style training；继续单 state 的 gain、anchor、solver 和小 adapter sweep 缺乏归因价值。
