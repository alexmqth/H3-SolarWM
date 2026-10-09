# 同容量FM16与AnyFlow16：完整评测均未通过

记录时间：2026-10-08T09:03:58.383670+08:00。两组全部39f/24fps A/D、4/8 steps/chunk已完成并可完整解码，所有conditioning检查通过。

在本轮16次更新、相同初始化/容量/数据/FP32与因果协议下，AnyFlow没有展示少步收益。4-step的FM明显比AnyFlow少雾化、人物与车库更完整，但FM仍有半透明/模糊；8-step两者都保留主要人物和场景，仍有拖影且A方向错误。不能推广为AnyFlow方法本身无效：这里只训练两条伪真值、16次更新，且本组两者均detach历史KV。

## 可播放的完整诊断视频

- [4 steps/chunk：Original / FM16 / AnyFlow16，A/D两行](matched_FM_AnyFlow_4step_AD_diagnostic.mp4)
- [8 steps/chunk：Original / FM16 / AnyFlow16，A/D两行](matched_FM_AnyFlow_8step_AD_diagnostic.mp4)

两条都保留完整39帧，不截掉末段；画面标注NOT PASSED、单次实测耗时与forward计数。它们属于诊断，不替换meeting主展示。

## A/D方向

| Method | steps/chunk | A flow | D flow | A-D | Gate |
|---|---:|---:|---:|---:|---|
| Original H3 | 30 full-seq | +1.181253 | -0.842124 | 2.023377 | reference |
| FM16 | 4 | -1.076402 | -1.290065 | 0.213663 | FAIL |
| FM16 | 8 | -1.135284 | -1.456185 | 0.320901 | FAIL |
| AnyFlow16 detached | 4 | -1.198536 | -1.242114 | 0.043578 | FAIL |
| AnyFlow16 detached | 8 | -1.305347 | -1.496471 | 0.191124 | FAIL |

严格gate仍为A>0、D<0、A-D>1与视觉完整同时成立；本轮所有A为负。

## 实测开销与连续性

| Method | Action | steps/chunk | E2E s | GPU allocated MiB | CPU KV MiB | noisy + commit | Gray MAD | Boundary RGB MAD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FM16 | A | 4 | 220.01 | 38924.02 | 6484.13 | 12+3 | 3.0970 | 3.6207 |
| FM16 | D | 4 | 222.26 | 38924.02 | 6484.13 | 12+3 | 3.1862 | 3.8846 |
| FM16 | A | 8 | 265.01 | 38924.02 | 6484.13 | 24+3 | 3.5440 | 4.3622 |
| FM16 | D | 8 | 257.72 | 38924.02 | 6484.13 | 24+3 | 3.5458 | 4.4734 |
| AnyFlow16 detached | A | 4 | 228.22 | 38984.16 | 6484.13 | 12+3 | 4.0724 | 6.0693 |
| AnyFlow16 detached | D | 4 | 220.69 | 38984.16 | 6484.13 | 12+3 | 4.0485 | 6.4854 |
| AnyFlow16 detached | A | 8 | 267.08 | 38984.16 | 6484.13 | 24+3 | 3.8701 | 5.2849 |
| AnyFlow16 detached | D | 8 | 271.64 | 38984.16 | 6484.13 | 24+3 | 3.7232 | 4.9133 |

GPU0/reserve6，单次测量，无warmup多次均值；旧teacher的offload reserve未记录，所以不据此宣布对原始H3的速度/显存收益。Gray MAD反映运动/像素变化，Boundary是RGB帧差，不能将两者直接相比或当作画质评分。

## 视觉复核范围与限制

FM A4/D4/A8/D8的0–38全部帧已通过全帧contact sheet查看，另逐项对比12/24/30/38帧的Original/FM/AnyFlow。FM四条未见单帧突发换场，4-step人物下半身与场景出现明显拖影和透明，8-step保留较多结构；这些是逐帧静态观察，不替代真人实时播放判断动作自然度。AnyFlow4后半段重影与结构雾化明显更重；8-step与FM没有明确优势。

## 训练控制

FM全部16次更新，含准备/验证1040.3037秒、allocated37970.60MiB。真实step00的visual/action/full-bank和CPU/CUDA/logical RNG与AnyFlow相同，16次action/chunk/sigma序列一致，四组LoRA都更新，visual/action冻结。FM不安装目标时间MLP；AnyFlow安装冻结的目标时间条件且loss有额外目标前向。因此是同容量/数据/更新次数的目标与采样对照，不是等计算量。

FM held-out weighted A：0.12627107→0.12617464，D：0.18172521→0.18161148，变化很小。不要与AnyFlow adaptive total直接比较。

下一步已按已证实的训练路径差异运行保留历史梯度的AnyFlow16；先看该实验，当前继续以Stage1效果验收为目标，Stage2暂缓。
