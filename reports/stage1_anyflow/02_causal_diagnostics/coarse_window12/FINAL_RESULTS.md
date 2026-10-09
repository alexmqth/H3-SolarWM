# E1 粗窗口完整结果：首窗改善，跨历史局部动作与结构仍 No-Go

2026-10-09T05:59:42.446689+08:00。Original H3 + released action LoRA，native text/action time，单 I0，30 steps，零训练。

## 决策

12-latent 当前窗口恢复了首窗口的方向，接入已知历史后仍未同时满足动作与人物结构门槛。停止继续增加窗口宽度，不用 AnyFlow 或 Stage2 处理尚未成立的局部正控。没有证据否定所有 causal 拓扑；这是本组零样本、有限场景下的负结果。

## 动作与画面

| 组 | 窗口 / RGB区间 | A flow | D flow | A−D | 逐帧评审 |
|---|---|---:|---:|---:|---|
| matched_first5 | 0 / [0,17) | -1.168911 | -1.174091 | 0.005179 | FAIL: A/D nearly identical |
| coarse_A | 0 / [0,39) | +0.863118 | -1.174217 | 2.037335 | 首窗方向恢复；人物/场景可辨 |
| coarse_A | 1 / [39,81) | +1.275624 | +1.176016 | 0.099608 | FAIL：同正方向，D肢体重影 |
| coarse_A | 2 / [81,120) | +1.342357 | -0.227267 | 1.569624 | 局部符号恢复；不能弥补前窗FAIL |
| coarse_D | 0 / [0,39) | +0.863568 | -1.174460 | 2.038028 | 首窗方向恢复；非独立历史证据 |
| coarse_D | 1 / [39,81) | +0.309645 | -1.482602 | 1.792246 | FAIL：符号正确，肢体重影 |
| coarse_D | 2 / [81,120) | -2.343244 | -2.324911 | -0.018332 | FAIL：A/D几乎相同且同负 |

flow 为既有 Farneback mean horizontal flow，单位 px/相邻帧。不是因果准确率；A>0、D<0只对本停车场正控使用，必须结合画面。MAD是活动量，不是画质。

两份reference的首窗都没有历史，不是两个独立泛化样本。[同前17RGB结果](FIRST_WINDOW_RESULTS.md)控制了统计长度，但12latent仍可使用整个当前已知窗口生成/解码，不代表相同控制延迟。所有分支共用history、初始noise、I0与布局哈希；A/D去噪轨迹随后分叉。本次没有把各自solver轨迹相减作同状态velocity证据，也没有完成新的same-state geometry。

人工评审：[完整记录](complete_visual_review.json)。检查完整静态帧和关键帧原尺寸细节，不冒称实时播放评审。MP4均完整解码核验。[固定A历史，当前A/D](coarse_A/AD_local_forks.mp4)；[固定D历史，当前A/D](coarse_D/AD_local_forks.mp4)。

这些120f视频是39+42+39帧的oracle局部串接，每窗恢复Original-generated参考历史；不是GT，也不是124f自由rollout。边界重置不作长时连续性证据；没有替换会议最终demo。

## 成本与执行核验

| 组 | noisy + diagnostic forwards | 全作业wall s | 模型加载 s | torch峰值allocated MiB | CPU hidden KV MiB |
|---|---:|---:|---:|---:|---:|
| matched_first5 | 60+3 | 246.90 | 3.44 | 26242.14 | 0 |
| coarse_A | 180+3 | 1878.81 | 3.43 | 26531.75 | 0 |
| coarse_D | 180+3 | 1881.43 | 3.40 | 26531.75 | 0 |

全作业wall包含同一进程内A/D多个分支、VAE、编解码/静态图与评测，不是一条视频的端到端推理耗时。最多3张项目GPU；共享主机、没有warmup后重复，不提供speedup。逐分支采样时间、frame MAD和raw-history大小见[完整数据](final_summary.json)。权重/激活峰值尚未单独分解。

T2每sigma重算可见history/prefix，CPU hidden KV=0，不能冒称persistent-KV加速；raw history latent另记。参数版本、历史hash、305文件runtime及启动源码/协议hash均保持。全部模型任务已退出，420 noisy+9 diagnostic forwards，optimizer=0。

## GT历史编码的未来RGB干预

| 验证片段 | RGB前缀/latent前缀 | 反转未来RGB后past latent最大差 | 仅过去RGB补末帧 vs 全片prefix最大差 |
|---|---:|---:|---:|
| 118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140 | 17/5 | 0 | 0 |
| 118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140 | 34/10 | 0 | 0 |
| 118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140 | 39/12 | 0 | 0 |
| 118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140 | 81/24 | 0 | 0 |
| dfec8ed3237860eba14d67c089ecd041_A_1030 | 17/5 | 0 | 0 |
| dfec8ed3237860eba14d67c089ecd041_A_1030 | 34/10 | 0 | 0 |
| dfec8ed3237860eba14d67c089ecd041_A_1030 | 39/12 | 0 | 0 |
| dfec8ed3237860eba14d67c089ecd041_A_1030 | 81/24 | 0 | 0 |

只加载VAE、无DiT/optimizer/下载。第一尝试D_1750原片剩余长度不足，加载模型前退出；[失败原记录](gt_boundary_audit/audit.json)保留。第二尝试换同验证episode的A_1030，纯RGB因果审计，不当作A/D配对。完整[审计](gt_boundary_audit_v2/audit.json)。干预结果仅对本布局/实现/两样本负责，不把尚未观察到的泄漏当失败根因；当前模型实验使用已有generated latent history，与GT编码审计不同。

## 下一步

区分已证实事实与待检验解释：首窗恢复、后窗失败已观察到；clean-history噪声/时间条件、位置、历史动作影响仍有混杂，尚未定位唯一根因。[CPU实际历史条件审计](history_contract_audit.json)的9状态确认历史video time=1，而全部text/action time=1−sigma；它符合现有pipeline/retake语义，不直接判bug。下一项[历史条件有限对照](NEXT_HISTORY_CONTROL.md)已设计、尚未实现/启动GPU；仍用Original权重、30步、固定12latent窗口/单I0，不扩大网络或窗口。

E2需要可靠的局部动作后果参照，不能用本轮失真的分支强行监督。E3仍为可信causal → AnyFlow → on-policy Stage2，不要求先解决20秒自由漂移，但首个局部验收尚未通过。
