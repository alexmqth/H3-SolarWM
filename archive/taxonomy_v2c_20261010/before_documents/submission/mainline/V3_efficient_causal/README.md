# V3 · Efficient Causal H3-World — 可行性版本

**2026-10-10，Judge正式验收。** 单停车场、seed13，AA/AD自身历史六块124帧（24fps，5.17秒）。同一个权重和推理协议同时具备strict chunk-causal、真实persistent KV、可辨动作响应与基本可用的多窗口人物/场景。画质和严格连续性仍有明显限制。

## 1. Version Name / Research Objective

V3 Efficient Causal H3-World，当前发布级别为feasibility。保留因果历史复用和动作条件能力，为后续少步方法提供可追溯基线。

## 2. Parent Version / Baseline

Original H3 + released action LoRA；从V2b借鉴native Single I0和12→5条件，复用旧current-prefix候选工程。没有继承V2a adapter、没有新训练checkpoint，也没有将V2b改名为V3。

## 3. Main Changes

V2b每个sigma对历史与当前双向重算；V3让当前video读取祖先raw KV和自己的action，当前action只反馈对应video，公共非action prefix只读当前video，每个chunk的prefix保持私有。祖先在自身生成时的图与native sigma0条件下提交一次，随后只读，未来action/video进入模型前物理删除。

## 4. Model and Inference Configuration

Original H3 + released action LoRA，SHA `ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526`。停车场832×480，seed13，相同full37 noise；Single I0、native时间、全局RoPE、固定audio条件；partition `[12,5,5,5,5,5]`、30steps/chunk、shift2.22、h3_fp32边界精度、原offload。CPU raw video KV，保留全部5个祖先；已显示RGB只追加不回改。0新增训练/AnyFlow/DMD。

首12是既有生成端点，空cache图/条件由EXP-002代码和真实模型identity检查支持复用。首39RGB为既有生成片的冻结像素，非GT；第二块起均接本候选自己的输出，未用teacher重置。

## 5. Representative Videos

- [Original vs V3 AA124](../../report/V3_efficient_causal/videos/Original_vs_candidate_AA_124.mp4)
- [V2b vs V3 AA124](../../report/V3_efficient_causal/videos/V2b_vs_candidate_AA_124.mp4)
- [V3 AA124原片](../../report/V3_efficient_causal/videos/AA_rollout_124.mp4) · [V3 AD124原片](../../report/V3_efficient_causal/videos/AD_rollout_124.mp4)
- [V2b/V3 AD共有73帧](../../report/V3_efficient_causal/videos/V2b_vs_candidate_AD_73.mp4)。Original A→D匹配参考缺失，不以持续D冒充。

## 6. Quantitative Results

| 新增RGB | AA flow | AD flow | AA / AD sampling秒 | AA对应V2b sampling秒 |
| --- | ---: | ---: | ---: | ---: |
| 73–89 | +1.463 | −1.304 | 148.45 / 148.94 | 310.26 |
| 90–106 | +0.953 | −0.648 | 176.43 / 176.66 | 377.53 |
| 107–123 | +1.335 | −1.240 | 200.42 / 199.43 | 447.19 |

同history第二块AA/AD响应由EXP-002验证（flow +1.347/−1.458）；后续两条历史已分化，不是同状态反事实。flow为辅助proxy。

EXP-003实际186forward/6VAE/0训练，0.436741GPU-hours；EXP-002+003合计310实际forward、10VAE、约0.66597GPU-hours，不包含复用首窗的原始生成开销。历史KV最终through32为18.131GB，50层各12480tokens。已记录新块peak26,876.70MiB；初始提交峰值具体值缺项但有≤44GiB断言。

对V2b相应AA块，候选single-run sampling约为44.8%–47.8%；这是两套协议整体观察成本，非纯KV单因素收益或公平完整E2E speedup。CPU→GPU搬运已计入sampling；完整首屏/从零E2E未测，30steps/chunk并非实时。

## 7. What Was Improved

同一明确配置已同时展示严格因果、真实持久缓存、局部同history动作控制与自身历史124帧基本可用性。历史Transformer不每步重算，实际增量sampling成本低于已有V2b对应块。

## 8. What Still Failed / Limitations

**不是全帧高质量：** AA约RGB79–86明显人体形变/游离肢体残影，第五、六块恢复；AA边界姿态跳变，AD91–92附近短暂姿态/肢体异常；动作僵硬、局部细节软化。严格连续性与画质成熟度PARTIAL，不能说追平V2b。没有跨scene/seed、10/20秒、其它动作序列或公平Original整体加速证明。

## 9. Lessons / Next Decision

有限正信号足以结束本轮可行性验证，不为几帧缺陷无限追加训练或消融。下一研究建议是在冻结V3基线上开展有预算的少步生成验证；AnyFlow/DMD仍未完成，新GPU任务另立任务书。

## 10. Source / Checkpoint / References

[EXP-002证据](../../experiments/EXP-002_native_cached/README.md) · [EXP-003源码/配置/manifest](../../experiments/EXP-003_native_cached_124/README.md) · [正式审核](../../experiments/EXP-003_native_cached_124/judge/FINAL_REVIEW.md)。正式EXP-003目录为`EXP-003_native_cached_124`；详见[来源和命令](../../experiments/EXP-003_native_cached_124/MANIFEST.md)。大型权重、latent和KV不提交Git。

## 11. 后续效率证据：EXP-004

[原权重8步续写AA/AD73帧](../../report/V3_8step_continuation/README.md)已获得有限正信号，零训练。首39帧仍借用30步生成结果，AA后段拖影明显；本增量证据不改变本页30步124帧版本定义。8步全程初始化、124帧与完整E2E未测，下一步优先解除30步前缀依赖。
