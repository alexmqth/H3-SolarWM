# 零更新：真实GT历史的局部30步基线

两条39帧局部片段已完成并逐帧静态检查；人物/场景大体保持，但起始边界跳变、姿态和真实后果差异仍在。不是action gate通过，也不是124帧自由rollout。

| 当前实际动作 | 采样秒 | 当前帧MAD | 边界MAD vs decoded GT |
|---|---:|---:|---:|
| W+A+J，F变化 | 445.15 | 19.250 | 23.376 |
| D+L+F | 451.77 | 12.425 | 23.757 |

两视频作业wall=971.21s，torch峰值allocated=25.62GiB；CPU hiddenKV=0，每sigma重算。耗时单次recorded run，无warmup多次均值，采样值不含VAE/加载，不是端到端加速结论。

- [村庄 W+A / camera-left](gt_history_baseline/118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015/GT_vs_local.mp4)：左真实GT，右Original+released action LoRA的T2/N。8GT context+39当前RGB，总47帧/24fps；source文件名的A/D只是旧first39标签，当前动作以这里及receipt为准。
- [中世纪 D / camera-right](gt_history_baseline/dfec8ed3237860eba14d67c089ecd041_A_1080/GT_vs_local.mp4)：左真实GT，右Original+released action LoRA的T2/N。8GT context+39当前RGB，总47帧/24fps；source文件名的A/D只是旧first39标签，当前动作以这里及receipt为准。

4个MP4完整H264/yuv420p/24fps解码通过。数据/梯度/编码与局部GT基线已齐，下一步只进入预注册4-update两臂目标比较。局部A/D反事实方向尚未通过，不启动AnyFlow或Stage2。
