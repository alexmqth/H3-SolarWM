# V0–V4统一5秒展示与完整生成成本（EXP-018）

**2026-10-13：执行准备中，尚无本轮实测结果。** 用户要求的统一对比已独立立项，7个冻结配置从噪声完整生成，0训练。旧分散计时不填入本轮结果表。

## 统一协议

停车场同一初图、scene prompt、持续A、seed13、832×480、24fps。H3原生长度5+17k，本轮每配置真实生成124帧（5.1667秒），保留原片并导出前120帧为严格5.000秒展示。耗时标注完整124帧，展示截取不改变统计范围。每版本保留其attention/anchor/adapter/chunk协议。

| 配置 | 采样设置 | 预定采样NFE（整段，不含commit） | 状态 |
| --- | --- | ---: | --- |
| V0 Original | 整片30 NFE | 30 | 待执行 |
| V1 Native causal | 8块 × 8 NFE | 64 | 待执行 |
| V2a RGB Anchor | 8块 × 8 NFE | 64 | 待执行 |
| V2b Same-σ | 6块 × 30 NFE | 180 | 待执行 |
| V3 Baseline | 6块 × 30 NFE | 180 | 待执行 |
| V3 FM-8NFE | 6块 × 8 NFE | 48 | 待执行 |
| V4 DMD-cycle8 | 6块 × 8 NFE | 48 | 待执行；旧模型曾画质失败，本次报告真实输出 |

从首窗重新生成，禁止使用旧endpoint/KV/首39帧充作本轮计算。V4从首窗评估属于独立扩展，不将旧56帧借首窗结果伪装为完整生成。124帧未在V3触发滑窗后续生成，不声称本轮检验无限长度。

## 将提交的统计口径

单行独立进程的模型/adapter加载、输入准备、采样、KV提交、解码、保存、ready-to-video和process-to-file wall分别记录；全流程allocated/reserved峰值（含初始化、采样、commit和解码），CPU KV另列。单次n=1，不声称多次统计稳定排名；precision/backend等必要版本差异明示。共享条件编码若采用则单独测量，fixture加载不当成编码开销。

每行提供124原片、120展示片、来源和权重SHA、日志与真实计数。质量/动作/连续性单独审核；数值有限但画质失败照实保留。完整成本与视频必须来自本轮运行，不拼接旧增量耗时。

[任务书和预算](../../experiments/EXP-018_unified_5s_benchmark/taskbook_v1.md) · [旧证据汇总](../original_vs_causal/README.md) · [报告首页](../README.md)
