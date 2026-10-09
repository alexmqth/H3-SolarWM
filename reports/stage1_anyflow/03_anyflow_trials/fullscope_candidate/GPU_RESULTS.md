# 全Q/K/V/out/FFN rank8：16次真实更新完成，视频评测开始

快照：2026-10-08T08:03:04.752819+08:00。GPU0独占、reserve6、CPU raw KV。当前子进程为`fullscope_fp32_step16_A_4step`（PID 3213786），首条A4尚未完成。

43,237,376训练参数，已完成总16次optimizer更新：1次fresh update，再精确恢复自身Adam/RNG到16。后者的脚本耗时1963.60秒，含准备和前后验证；第一段为451.63秒，两段合计2415.23秒，合计包含两次启动/验证，不能当作不中断训练的测量。续训allocated峰值37386.21MiB（36.51GiB），未OOM；nvidia-smi与PyTorch allocated计量不同。

QKV/out/FFN/refiner四组均实际改变，原visual adapter和目标时间MLP严格不变。初始固定noise验证、312个逻辑投影B因子更新及GPU精确恢复的证据保留在本目录的audits。

| 相同noise/时间的验证项 | 初始 | full-scope 1 | full-scope 16 | native tail16 /16 |
|---|---:|---:|---:|---:|
| A weighted | 0.118052 | 0.118067 | 0.118061 | 0.117049 |
| A endpoint raw | 59.624653 | 69.482544 | 71.848572 | 55.934113 |
| A flow-map raw | 0.168970 | 0.169805 | 0.170680 | 0.167230 |
| D weighted | 0.170266 | 0.170327 | 0.170215 | 0.169150 |
| D endpoint raw | 22.804770 | 21.892948 | 22.714802 | 22.690615 |
| D flow-map raw | 0.228104 | 0.227544 | 0.226449 | 0.224058 |

A endpoint较初始变差，D略降；weighted与general-map变化很小。当前没有明确的固定噪声验证收益，不能据此宣称扩大训练范围解决了AnyFlow。16次仍是小规模pilot，与官方的数据/批量/步数差距很大；这些结果也不能证明全覆盖AnyFlow经过充分训练仍不可行。

已开始A/D39f的4/8 steps/chunk评测。需要完整视频、方向/分离度及场景/人物检查；视频未完成前不提前填action或视觉结果。同容量native-FP32 full-scope FM16控制器已在等待，完成后才能进一步区分训练容量和AnyFlow目标/条件/采样的作用。

可选全覆盖bank与FM评测已合入主代码和submission，默认tail路径不变；主代码67 tests passed。当前运行的冻结runtime未改。Stage1效果gate仍未通过，不替换会议视频，不启动Stage2。

## 首条A4评测（2026-10-08T08:08:17.133866+08:00）

完整39帧已解码，same image/prompt/action/seed/video+audio noise逐张量一致；实际载入43,237,376参数的stage1_lora bank。A flow=-1.198536，符号错误；完整39帧图显示约20帧开始严重重影/雾化，后段人物和车库结构丢失，与native tail16无明确修复。动作/视觉均未通过。

E2E=228.22s，GPU allocated peak=38984.16MiB，CPU KV=6484.13MiB，12 noisy forwards+3 clean commits。单次本机记录，不作公平speedup结论。

见`fullscope16_A4_all39.jpg`和`fullscope16_A4_vs_teacher_tail.jpg`；当前D4/A8/D8仍待完成。
