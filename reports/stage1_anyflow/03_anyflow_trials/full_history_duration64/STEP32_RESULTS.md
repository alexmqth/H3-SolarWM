# Full-history AnyFlow32：完整短片未过动作验收

记录时间：2026-10-08T11:09:23.838934+08:00。本报告只评估step32；step64仍在训练。

[完整39帧：Original / AnyFlow16 / AnyFlow32，A/D两行](original_anyflow16_anyflow32_AD.mp4)

## 动作结果

| Method | steps/chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
| Original H3 | 30 full sequence | +1.181253 | −0.842124 | 2.023377 |
| full-history AnyFlow16 | 8 | −1.304568 | −1.487077 | 0.182509 |
| full-history AnyFlow32 | 8 | -1.309516 | -1.497935 | 0.188419 |

A仍为负，不满足A>0、D<0、A−D>1。分离度变化约0.006，不是动作恢复。

| Action | E2E s | allocated GPU MiB | CPU KV MiB | noisy + commits | Gray MAD | Boundary RGB MAD |
|---|---:|---:|---:|---:|---:|---:|
| A | 260.76 | 38984.16 | 6484.13 | 24+3 | 3.851934 | 5.211747 |
| D | 266.31 | 38984.16 | 6484.13 | 24+3 | 3.721822 | 4.836029 |

两条全部0–38帧contact sheet及Original/16/32在12/24/30/38帧的对照已审阅。人物/车库主体保留，但后段仍模糊、重影，D的人物有透明感；与16次没有明确可见改善。这里是静态全帧与同帧检查，不声称完成真人实时播放评审。视频保留所有39帧，H.264/yuv420p/24fps，已完整解码验证。

保持full-history、train/validation/inference shift2.22、rank8全覆盖、冻结visual/action/time、RGB dual、CPU KV、seed13；只增加更新数。两条conditioning都与原始H3逐张量一致。训练曲线见[GPU_RESULTS.md](GPU_RESULTS.md)：A endpoint变小但D变大，不能据此预判画质。

E2E是与GPU1/GPU3训练并行时的单次记录，不是warmup多次均值，也不用于宣称speedup。MAD不是画质分数，gray与RGB边界统计口径不同。

此分支已按预定规则从step32续到总64；保留原Adam与RNG，不加架构/anchor/LR变量。GPU1 shift12续训和GPU3匹配full-history/shift12/FM32同时运行。FM32应与shift12 AnyFlow32比较；同更新数不等计算预算。当前Stage1未通过，Stage2暂缓。

64续训的真实回载快照逐项验证通过：[gpu_resume32_audit.json](gpu_resume32_audit.json)。这确认保存state精确恢复，不代表独立CUDA训练梯度逐bit可复现。
