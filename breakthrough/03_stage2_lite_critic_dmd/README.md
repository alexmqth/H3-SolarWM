# Breakthrough 03：Stage2-lite critic/DMD 链路跑通

## 问题

per-sigma teacher replay 或简单 two-pass replay 只能把 student 拉向 teacher score，不能显式表示 student 自己的 generated distribution。需要加入 SolarWM Stage2 的关键角色：trainable fake-score critic。

## 方案

使用同一个 33B H3 backbone，轮换三种 adapter：student causal action residual、critic/fake-score residual、teacher frozen path。流程是：

    student self-rollout -> detach history -> critic fit -> frozen teacher -> DMD surrogate

本项目从 39 frames、3 chunks、chunk 1/2、4 sigmas、1 update 开始；KV 在 CPU，RGB anchor 固定，避免把视觉 anchor 和 Stage2 loss 同时改变。

## 证据

`evidence_stage2_lite_AD_39.mp4` 是早期链路视频；`evidence_integrated_stage2_lite_AD_39.mp4` 是 RGB-consistent visual adapter 接入真实 critic/DMD chain 后的结果。训练完成且 finite，无 NaN/OOM，单张 L40 可运行。

## 结果

integrated run 的 A/D flow 为 A=-1.1419、D=-1.4550、A-D=0.3131；人物和停车场结构稳定，但严格 action gate 失败。

## 结论

Stage2-lite 的工程链路可运行，且不再引入旧的人物分解；但一轮小规模 DMD surrogate 没有恢复 action geometry，也不能被称为官方 Stage2 或有效少步模型。
