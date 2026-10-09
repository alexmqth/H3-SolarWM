# V1 · Native Chunk-Causal H3-World

## 1. Version Name / Research Objective

验证 Original 权重能否按chunk生成并使用真正的persistent raw KV，展示直接因果化带来的动作/画面问题。

## 2. Parent Version / Baseline

V0权重。主代表明确选零新增adapter的native后期latent-dual协议，不是fixed-mix，也不是最早seed2版本。

## 3. Main Changes

5 latent/chunk，generated history，CPU raw video KV，clean commit和5-chunk窗口；dynamic_last_frame_dual。own action prefix / feedback off；固定prefix时间，与原生时间不同。

## 4. Model and Inference Configuration

2026-10-02 action_{A,D}_base_causal124_retry2；124f/24fps，seed13，8steps/chunk，shift2.22，64 noisy forwards + 8 clean commits；trained_for_causal=false，causal_adapter=null。

## 5. Representative Videos

- [Original vs V1，A/D完整124f](videos/V1_vs_original.mp4)
- [V0→V1；与前一文件字节相同的命名别名](videos/V1_vs_V0.mp4)
- [V1 A原片](videos/V1_A_full.mp4)
- [V1 D原片](videos/V1_D_full.mp4)

## 6. Quantitative Results

原记录A=−0.017959、D=−0.025755，A−D=0.007796；A/D E2E分别465.9/400.9s，CPU KV13.19GiB，GPU allocated peak约39.01GiB。这些为共享硬件单次记录。

## 7. What Was Improved

逐块执行、clean commit、历史复用和完整124帧输出已实现。更早seed2/4-step版本另存，不以新配置改写早期证据。

## 8. What Still Failed

本代表A/D运动几乎消失、A符号错；抽帧显示人物停滞、后段过亮和背景退化。更严重的人物撕裂/重影见另存的fixed-mix及早期DMD片，不能归成同一个checkpoint。

## 9. Lessons Learned

Causalization是协议改造，不自动保留Original的动作传播。persistent KV执行成功也不等于动作或质量成功。

## 10. Source Code / Checkpoint / Original Experiment References

H3-World/outputs/2026-10-02-04/action_{A,D}_base_causal124_retry2/。无新增训练checkpoint；历史原始底座与released LoRA仍为外部依赖。

[核心代码说明](code/README.md) · [代码SHA与源路径](code/SOURCE_MANIFEST.json) · [本版来源清单](PROVENANCE.json) · [返回汇报导航](../README.md)

指标均为历史记录；flow是运动proxy，MAD是活动量/连续性描述，不是视频质量评分。Original生成视频不是GT。
