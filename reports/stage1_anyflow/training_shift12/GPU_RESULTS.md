# Train shift12：16次更新与四条视频已完成

完整视频结果见[FINAL_RESULTS.md](FINAL_RESULTS.md)，下文保留训练完成时的快照。

训练含初始化/前后验证 2516.05s；GPU allocated峰值 41310.95MiB。训练shift12，公共validation/inference保持2.22。原始四套adapter、CPU/CUDA/logical RNG、初始validation、实际全部16次sigma/r与额外历史前向次数审计通过。

| Action | 公共验证项目 | step00 | shift2.22 step16 | shift12 step16 |
|---|---|---:|---:|---:|
| A | weighted total | 0.118051555 | 0.117942682 | 0.118034674 |
| A | diffusion raw, sigma=0.59218 | 0.064436495 | 0.064374566 | 0.064428739 |
| A | diffusion raw, sigma=0.29619 | 0.140979081 | 0.140851796 | 0.140957251 |
| A | endpoint raw, sigma=0.92971 | 59.624652863 | 61.344249725 | 54.556304932 |
| A | flow_map raw, sigma=0.30200 | 0.168970197 | 0.172369286 | 0.170321628 |
| D | weighted total | 0.170266438 | 0.170181701 | 0.170265539 |
| D | diffusion raw, sigma=0.59218 | 0.096876085 | 0.096801162 | 0.096842855 |
| D | diffusion raw, sigma=0.29619 | 0.198909059 | 0.198840111 | 0.198944777 |
| D | endpoint raw, sigma=0.92971 | 22.804769516 | 22.865306854 | 24.329162598 |
| D | flow_map raw, sigma=0.30200 | 0.228104293 | 0.225356147 | 0.226440981 |

A endpoint59.625→54.556，有下降；D22.805→24.329，上升。普通diffusion raw变化很小，weighted total也几乎不变。结果混合，不等于少步生成已改善。AnyFlow自适应缩放使weighted endpoint几乎保持不变，所以必须同时看raw residual。

当前GPU1自动执行A/D39f、4/8 steps/chunk generated-history评测；GPU3只读同一step16做A/D8 clean-history诊断。后者不能作为free-running验收，需与前者一起解释。GPU0另从shift2.22/full-history16精确续训到32→最多64，不能把其训练量变化归到shift12实验。

当前Stage1动作与画面gate尚未通过，Stage2继续暂缓。独立GPU反传有轻微重复性误差；单次小残差差异不构成严格的效果因果证明。
