# 停车场8步：A/D近同向，后段重影

这四条停车场causal0评测均为新增adapter零更新，不是FM48效果。8-step/chunk的A/D全部39帧contact sheet与0/8/16/24/30/38对应帧已静态检查：前段人物可辨，约20–22帧开始明显透明/多轮廓，22–38帧持续重影，车库地面/柱子出现叠影。A、D画面及运动非常接近。未作人工实时播放。

A=-0.021397，D=-0.031908，A−D=0.010510；Original30 A=+1.181253、D=-0.842124、分离度2.023377。数值与视觉均未通过。30-step/chunk的画面比8步完整，但A−D=-0.008565，同样没有恢复动作区分；因此画质和控制必须分别记录。

每条8步为24 noisy forwards+3 clean commits，CPU KV6484.13MiB、allocated peak16234.51MiB；A/D单次recorded wall302.15/299.08秒。初始noise/audio/prompt/anchor/action rows与各自Original逐张量相同；Original legacy精度、causal0 h3_fp32与prefix/anchor差异披露在README，不宣称单一mask归因、公平加速或显存节省。

[完整视频](original_causal_AD.mp4) · [指标CSV](metrics.csv) · [A对应帧](A_matched.jpg) · [D对应帧](D_matched.jpg)。FM48训练后按同设置比较，不根据零更新结果提前判训练成败。
