# EXP-004 / v1 Worker 执行报告：V3 原权重 8-step native FM 续写

更新：2026-10-10 HKT。**Worker Status: completed；Judge Acceptance: pending。** 本轮按 [EXP-004 任务书](next_plan.md)执行，完整证据在 [独立实验目录](submission/experiments/EXP-004_v3_8step/README.md)。GPU 进程已全部退出；没有启动训练、AnyFlow 或 DMD，也没有超出预算的推理。

## 研究问题与结论

只将 V3 新增 chunk 的 native FM/Euler sampling 从30步改为8步，是否仍保留同历史 A/D 条件响应，并能接自己的8步历史续至73帧？**本轮有限单场景/seed实验得到正的可行性信号，但画质与动作幅度不完全保真。** AA/AD第二块共用 EXP-002 保存的首12 latent、首39 RGB、首窗 CPU raw KV、noise、anchor、audio、全局位置、权重与推理协议，只改变当前动作；8步水平光流分别为 **+0.781 / −1.510 px/帧**，方向相反。两条路径各自接自己的8步第二块后，第三块为 **+0.978 / −0.732 px/帧**，人物和停车场到73帧仍可辨。第三块两侧的历史已不同，不能当作同状态反事实。

首块是**复用的30-step**结果，只有39–55和56–72两个新增RGB区间对应8-step采样。因此本轮不能声称完整73帧从零8-step、8-step首屏延迟或完整E2E加速。直接减步用的是原生FM，不是AnyFlow目标或蒸馏。

| 路径 / 新增RGB | 本轮8步flow | 已存V3 30步flow | 本轮sampling / 30步sampling | 本轮边界灰度MAD | 视觉判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| AA 39–55 | +0.781 | +1.347 | 70.87 / 142.91 s | 3.54 | 人物单体、场景可辨；动作幅度较小、细节更软 |
| AD 39–55 | −1.510 | −1.458 | 40.52 / 135.69 s | 2.81 | 人物单体、方向可辨，普通腿部细节缺陷 |
| AA 56–72 | +0.978 | +0.474 | 37.76 / 155.14 s | 4.89 | 仍可辨，但腿部拖影、局部透明残影较明显 |
| AD 56–72 | −0.732 | −0.930 | 42.51 / 146.39 s | 7.95 | 仍可辨，块边界变化与局部腿纹理残影 |

30步实验在清晨、本轮在下午共享硬件有其他作业；上表是**各自一次运行的实际sampling观测**，不能直接计算公平的速度倍数。flow是运动代理，不能单独代替动作语义或画质判断。AA第三块与30步出现不同的人物朝向和运动轨迹；不能把两侧不同历史的第三块差异归因于单步velocity变化。

## 视频、协议与核验

- [AA 56帧：左V3 30步、右8步](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AA_56.mp4)；[AD 56帧同历史对照](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AD_56.mp4)。
- [AA 73帧闭环对照](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AA_73.mp4)；[AD 73帧闭环对照](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AD_73.mp4)。四条8步原片也保存在同一 [视频目录](submission/experiments/EXP-004_v3_8step/artifacts/videos/)。全部新帧与必要原分辨率细节已查看，坏帧未裁除。
- 原生8步sigma为 `1, 0.939540, 0.869452, 0.787234, 0.689441, 0.571184, 0.425287, 0.240781, 0`；没有截断30步sigma。Second chunk采样只读原始首窗KV；进入third前只将本轮自己的8步second clean endpoint在sigma=0提交一次；third无多余commit。future action/video在模型前物理裁剪，保留accepted current-prefix own-action路由、Single I0与全局RoPE。
- 四段源代码和条件摘要一致；第二块AA/AD的首窗latent、首窗KV、初始noise、anchor、audio、位置、sigma和旧RGB摘要相同，仅prompt动作摘要不同。采样前后历史cache的entry identity、storage address、tensor version与commit计数未变；RGB前39及随后前56帧在`.npy`层逐像素冻结。四原片与四并排片均通过H.264、24fps、帧数、尺寸、单调PTS和完整解码检查。详见 [来源与命令](submission/experiments/EXP-004_v3_8step/MANIFEST.md)、[视频审计](submission/experiments/EXP-004_v3_8step/artifacts/video_integrity.json)、[机器指标](submission/experiments/EXP-004_v3_8step/metrics.json)。

## 预算、成本与边界

实际 **32 sampling + 2 clean commit = 34次完整denoiser forward、4次VAE decode、0 optimizer update**。GPU 0顺序执行四个进程，共 **466.02 GPU-seconds = 0.12945 GPU-hours**，低于0.5 GPU-hours与90分钟elapsed上限；没有失败/自动重试。记录到的单进程最大 `torch.cuda.max_memory_allocated` **25,682.07 MiB**，第二/第三块采样前的CPU raw KV分别约 **6.799 / 9.632 GB**。commit约12.02/12.19秒；这些测量包含真实cache使用，sampling内的搬运未单独拆分。进程增量wall和采样时间见 [逐块账本](submission/experiments/EXP-004_v3_8step/artifacts/metrics/)；不包含首39帧旧生成成本。

Worker判断：执行协议有效，同历史第二块A/D动作差异和自身历史第三块结构有明确有限正结果；本轮8步画质及AA动作幅度较30步下降，AA后段残影与AD边界变化需保留。因此建议将**原权重直接8步的局部能力**标为 `PARTIAL`，由Judge独立决定是否接受为后续少步研究基线。尚未验证124帧、其它场景/seed、动作切换或完整端到端效率；不自动扩展至这些范围，也不把这项结果等同AnyFlow/Stage2。
