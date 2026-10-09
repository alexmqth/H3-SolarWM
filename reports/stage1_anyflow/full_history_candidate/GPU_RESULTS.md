# Full-history AnyFlow16：训练与step16四条评测完成

完整结果见[FINAL_RESULTS.md](FINAL_RESULTS.md)，以下保留训练及首条视频时的历史快照。

快照：2026-10-08T09:52:03.113075+08:00。GPU0/reserve6完成16次更新；包含初始化和前后验证耗时2863.15秒，allocated峰值41310.60MiB。原visual/time仍冻结，QKV/out/FFN/refiner四组LoRA实际更新。

实际step00四套adapter、CPU/CUDA/logical RNG、初始公共validation samples与detached对照相同，全部训练action/chunk/sigma匹配。每次full-history前向次数与chunk_index×logical_batch一致；合计额外56次带梯度历史前向。独立GPU反传有微小重复性误差，不能把很小的优化轨迹差异完全归因于历史梯度。

| Action | 项目 | step00 | step16 |
|---|---|---:|---:|
| A | weighted total | 0.118051555 | 0.117942682 |
| A | diffusion raw, sigma=0.59218 | 0.064436495 | 0.064374566 |
| A | diffusion raw, sigma=0.29619 | 0.140979081 | 0.140851796 |
| A | endpoint raw, sigma=0.92971 | 59.624652863 | 61.344249725 |
| A | flow_map raw, sigma=0.30200 | 0.168970197 | 0.172369286 |
| D | weighted total | 0.170266438 | 0.170181701 |
| D | diffusion raw, sigma=0.59218 | 0.096876085 | 0.096801162 |
| D | diffusion raw, sigma=0.29619 | 0.198909059 | 0.198840111 |
| D | endpoint raw, sigma=0.92971 | 22.804769516 | 22.865306854 |
| D | flow_map raw, sigma=0.30200 | 0.228104293 | 0.225356147 |

两条weighted total下降不足0.1%；A endpoint59.625→61.344、D22.805→22.865，未见明确端点改善。不能据此宣称少步画质改善，仍需完整A/D4/8视频。

先前step04 A/D8已失败，A-D=0.185948；它是不同更新数的早期检查，不是full16最终结果。视频和全帧复核见[step04结果](../full_history_step04/FINAL_RESULTS.md)。

当前队列：full_history_fp32_step16_D_4step；训练分布shift12的独立16次对照同时在GPU1运行，验证和推理仍2.22。两条路线都未通过Stage1 gate，Stage2暂缓。

## 首条step16 A4视频（2026-10-08T09:58:01.024810+08:00）

A flow=-1.248505，方向gate失败。完整0–38帧及与Original/FM16/detached16同帧检查显示，约20帧后出现严重重影和雾化，人物/车库结构丢失；相对detached16没有明确修复，FM4更完整。此条没有显示补齐历史梯度后的4-step收益。D4/A8/D8尚待完成，不能提前写成全部已评测。

[同帧对照](full16_A_4step_comparison.jpg)；[全39帧](full16_A_4step_all39.jpg)。
