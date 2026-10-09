# 同checkpoint的teacher/generated history：仍未达到动作gate

[完整39帧：Original / generated history / teacher history，A/D两行](original_generated_clean_history_AD.mp4)

两列Stage1使用相同shift12 step16权重、image/prompt/action/seed13/video-audio noise、8 steps/chunk、RGB dual和CPU raw KV。唯一差别是后续commit及RGB anchor来源于teacher还是模型自身。右列明确标为TEACHER history，不能当free-running效果。

| History | A flow | D flow | A−D |
|---|---:|---:|---:|
| generated | -1.312275 | -1.492164 | 0.179888 |
| teacher (oracle) | -0.211312 | -0.875811 | 0.664499 |

A/D第一块5个latent在teacher/generated两种运行中均逐元素相同（max_abs=0，分别见first_chunk_A_audit.json与first_chunk_D_audit.json）；之后latent才不同。不是换checkpoint或随机noise带来的对照差异。

## 逐帧观察与边界

两条teacher-history的全0–38帧已通过contact sheet复核，并与Original及同checkpoint的generated-history比较12/24/30/38帧。teacher历史让后续场景更接近参考，但A在17/34帧附近人物位置/朝向明显重置，D也有块边界状态跳变与模糊；这些拼接不连续不能作为生成能力改善。

| History | Action | E2E s | Gray MAD | Boundary RGB MAD |
|---|---|---:|---:|---:|
| generated | A | 268.77 | 3.8185 | 5.3323 |
| generated | D | 264.12 | 3.6973 | 4.8124 |
| teacher | A | 224.76 | 4.5057 | 14.8227 |
| teacher | D | 253.19 | 3.9266 | 9.2856 |

使用已有Farneback segment_flow复核A：第0块[0,17)原始H3 flow=+0.083398，而teacher-history诊断为−1.207126；后续排除切换边界后的[18,34)/[35,39)分别+0.223086/+1.376205。原始对应为+2.220752/+1.570233。完整分段见segment_A8.json；像素光流仍是运动代理，不是严格action accuracy。

该结果同时说明两件事：teacher history能约束后续状态；第一个无生成历史的chunk也已出现动作偏离，当前不足不能全归因于generated-history累积误差。teacher历史本身含action后的场景状态，后续正flow并不独立证明模型当前action binding正确。

未通过A>0、D<0、A−D>1，且oracle reset违反连续性。保持Stage1训练量对照，不据此提前进入Stage2或扩124帧。两条视频24 noisy forwards+3 commits；耗时为多卡并行单次记录，不能据此解释速度差。
