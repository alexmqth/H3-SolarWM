# 主要历史指标（不新增模型测量）

统一摘录原收据，保留作用范围。下面是共享主机单次记录，非统一offload/warmup多次均值，不能直接排名speedup。

| 版本/动作 | 范围 | 时间s / scope | GPU peak GiB | CPU KV GiB | noisy+commit | flow x |
|---|---|---:|---:|---:|---|---:|
| V0 A | 124f全片 | 454.2 E2E | 38.99 | 0.00 | 30+0 | +1.0767 |
| V0 D | 124f全片 | 450.4 E2E | 38.99 | 0.00 | 30+0 | -1.6019 |
| V1 A | 124f全片 | 465.9 E2E | 39.01 | 13.19 | 64+8 | -0.0180 |
| V1 D | 124f全片 | 400.9 E2E | 39.01 | 13.19 | 64+8 | -0.0258 |
| V2a A | 124f全片 | 699.5 E2E | 39.00 | 13.19 | 64+8 | -0.7841 |
| V2a D | 124f全片 | 715.9 E2E | 39.00 | 13.19 | 64+8 | -1.0075 |
| V2b A→A | 第二块17f | 195.5 sampling | 25.42 | 0 | 30+0 | +2.1464 |
| V2b A→D | 第二块17f | 195.3 sampling | 25.42 | 0 | 30+0 | -1.7519 |
| V2b D→A | 第二块17f | 194.8 sampling | 25.42 | 0 | 30+0 | +2.6076 |
| V2b D→D | 第二块17f | 194.6 sampling | 25.42 | 0 | 30+0 | -1.4143 |

V2b复用首39帧，不把第二块sampling冒充全片E2E；有些诊断forward时间另记。CPU KV=0不表示CPU模型权重/状态为0。GPU为torch allocated峰值，不是整卡占用，也未分解为权重/KV/激活。

8steps/chunk×8chunks=64 noisy forwards，再加8clean commits；Original30整段forwards每次序列更长。步数30/8不是端到端加速比。首块latent计时也不等于用户能看到首屏：原benchmark末尾统一decode。

flow是Farneback中央裁剪水平位移proxy，A正/D负须与人物和场景一起看；不能评判W/S前后移动。V2b是17新RGB段，不能与124全段flow比较恢复百分比。MAD低可能只是画面模糊/静止，不是更高质量。未计算FVD/LPIPS/PSNR/VBench，Original也不是GT。

原始收据：V0/V2a `meeting/source_metrics/rgb_visual/summary.json`；V1 `action_base_causal124_flow.json`与每运行cached.json；V2b `experiments/11_causal_12_then5_selfhistory/summary.json`。各版PROVENANCE保存选中运行完整指标/配置。

[返回首页](README.md)

## 原有连续性指标（不同评估帧段不作排名）

| 版本/动作 | 帧段 | Frame RGB MAD | Boundary RGB MAD |
|---|---|---:|---:|
| V0 A | 124f | 4.5199 | 5.1423 |
| V0 D | 124f | 4.4583 | 5.3894 |
| V2a A | 124f | 3.1855 | 3.7871 |
| V2a D | 124f | 3.0356 | 3.8874 |
| V1 A/D | 124f | 未找到同口径原记录 | 未找到同口径原记录 |
| V2b A→A | 当前17f / 38→39边界 | 5.8999 | 4.3836 |
| V2b A→D | 当前17f / 38→39边界 | 5.2963 | 6.4871 |
| V2b D→A | 当前17f / 38→39边界 | 6.6188 | 4.8508 |
| V2b D→D | 当前17f / 38→39边界 | 3.9685 | 3.8542 |

V0/V2a边界为旧chunk5协议的多个边界均值；V2b边界为38→39的一次转场。V1缺项没有填0，也没有把另一个checkpoint指标移植过来。本次未新增质量测量。
