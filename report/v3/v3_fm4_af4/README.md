# V3 4NFE对照：普通FM4与AnyFlow AF4

**EXP-009已完成：普通FM4 AA/AD73帧有限续写可用、quality PARTIAL；AF4 AD第三块持续人物分解，生成FAIL，当前AF4配置归档。** 固定AF2 step32在4NFE没有显示整体优于普通FM4的收益；没有追加训练或扫描更低步数。

两候选共享EXP-006 FM8首12 latent/首39 RGB，各自权重建立clean KV。FM4使用Original H3 + released Action LoRA及原生4步scheduler；AF4加载新AF2 step32 QKV/target-time，用next-sigma finite map。Global/current-prefix/own-action/strict causal/persistent KV保持。C2同历史，C3各自生成历史。**这证明的是4NFE续写，不包含4步首窗生成，也不证明长视频或滑窗淘汰。**

| 配置 | C2 AA / AD水平flow | C3 AA / AD水平flow | 视觉与控制 |
| --- | --- | --- | --- |
| 普通FM4 | +.706 / −.190 | +.336 / −.618 | 人物/场景可辨，A/D方向可见；腿部残影与边界跳变，PARTIAL。 |
| AF4 step32 | +.489 / +.162 | +.268 / −.0075 | AA透明化加重；AD C2扭曲、C3人物持续碎裂重影且响应近消失，该分支FAIL。 |

判断基于全部新增帧和原分辨率细节，光流只辅助。AF4失败不改写AF8已有有限可行性，也不能推广为所有AnyFlow训练失败。

[Judge正式审核](../../../experiments/EXP-009_v3_fm4_af4/judge/FINAL_REVIEW.md) · [机器审计](../../../experiments/EXP-009_v3_fm4_af4/judge/c3_audit.json) · [逐段视觉记录](../../../experiments/EXP-009_v3_fm4_af4/judge/visual_notes.json) · [实验入口与视频](../../../experiments/EXP-009_v3_fm4_af4/README.md)

实际38forward（32sampling+6clean commit）、8VAE、0训练，782.290秒=.217303GPUh，allocated峰26.336GiB。单GPU0顺序执行，无重试；每块sampling约22.5–24.9秒，decode约4.9–6.8秒。重复加载/保存/缓存I/O另计，不以NFE减半宣称公平E2E加速。

下一主线EXP-010准备把已有全程FM8扩展到124/158帧，检验与真实滑窗淘汰的组合。正式30步V3/SW-G结果冻结。

[AA73四宫格视频](../../../experiments/EXP-009_v3_fm4_af4/artifacts/comparisons/AA_FM4_AF4_with_8NFE_73.mp4) · [AD73四宫格视频](../../../experiments/EXP-009_v3_fm4_af4/artifacts/comparisons/AD_FM4_AF4_with_8NFE_73.mp4) · [Worker完整报告](../../../experiments/EXP-009_v3_fm4_af4/worker_report_final.md)
