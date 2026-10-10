# EXP-003：V3 Baseline Strict Causal KV · 124帧验证

**当前研究分类：V3 Baseline Strict Causal + Persistent KV。** 原Efficient Causal路线曾称V3；历史任务/报告/指标/源码保留原称，现行分类见[V2家族](../../report/v2/README.md)。

**Judge accepted：同一严格缓存协议的124帧可行性通过，发布为V3 Baseline可行性版本。** [正式审核](judge/FINAL_REVIEW.md)。 本轮从 [EXP-002 已验收的73帧候选](../EXP-002_native_cached/README.md)继续，不重建祖先 KV：AA/AD 各加载自己的 `first12_A + second5 + third5` 生成 endpoint 与 `cache_through17`，先将第三块在原生条件下 sigma=0 clean commit 一次，然后依次生成 `[22,27)`、`[27,32)`、`[32,37)`，得到90、107、124 RGB帧。Original H3 + released action LoRA、Single I0、current-prefix own-action 路由、全局位置、30-step/chunk、shift 2.22、固定 audio、CPU raw KV 和 seed 13 均保持不变。没有训练、GT reset、AnyFlow、DMD 或最后块多余 commit。

| 新增 RGB | AA 水平 flow | AD 水平 flow | AA/AD 边界灰度 MAD | 视觉观察 |
| --- | ---: | ---: | ---: | --- |
| 73–89 | +1.463 | −1.304 | 13.82 / 7.18 | **AA约79–86帧出现明显人体形变和游离肢体残影**；AD人物基本完整 |
| 90–106 | +0.953 | −0.648 | 9.69 / 3.34 | AA恢复单体；AD约91帧短暂肢体残影 |
| 107–123 | +1.335 | −1.240 | 2.91 / 2.50 | 两条人物与停车场均可辨；AA约108–110帧短暂背景残影 |

水平光流是方向代理，不能替代动作语义评判。AA 的第四块有实质视觉失败片段，后两块没有持续累积崩坏；因此结论是**单场景/seed 下有质量限制的124帧可行性**，不是“全程稳定高质量”。第二块的同历史 A/D 反事实能力证据来自 EXP-002；本轮从73帧起两条路径拥有不同的自身历史，不再是同状态反事实。

主要视频（均为24fps、真实 H.264）：

| 内容 | 链接 |
| --- | --- |
| 完整候选 | [AA 124帧](artifacts/videos/AA_rollout_124.mp4) · [AD 124帧](artifacts/videos/AD_rollout_124.mp4) |
| 逐块留存 | [AA 90](artifacts/videos/AA_rollout_90.mp4) · [AA 107](artifacts/videos/AA_rollout_107.mp4) · [AD 90](artifacts/videos/AD_rollout_90.mp4) · [AD 107](artifacts/videos/AD_rollout_107.mp4) |
| 左 Original、右候选 | [Original A vs candidate AA 124](artifacts/videos/Original_vs_candidate_AA_124.mp4) |
| 左 V2b、右候选 | [V2b AA vs candidate AA 124](artifacts/videos/V2b_vs_candidate_AA_124.mp4) · [V2b AD vs candidate AD 共73帧](artifacts/videos/V2b_vs_candidate_AD_73.mp4) |

Original A→D 匹配视频不存在，本轮没有用 Original 持续 D 冒充，也没有新增基线推理。Original AA 124、V2b AA 124 与候选 AA 124 的权重输入来源及协议差异见 [比较 manifest](artifacts/comparison_manifest.json)。这些并排片是能力比较，不是单因素消融；视频标明 Original 整片耗时与候选73→124帧增量耗时，不能计算统一的端到端加速比。

工程证据：[显式区间入口](interval_cached.py)支持完整12+5×5分块，[计时路由](metered_prefix.py)只在已验收 current-prefix attention 中增加KV读取与传输事件，不改 attention 输出；与原路由CPU输出逐元素一致。[11项CPU协议测试](artifacts/cpu_tests_postrun.log)通过，原始 endpoint/cache/hash [预检日志](preflight_cpu.log)通过。六个新块的采样均验证内存历史 cache entry/storage/version 不变；提交后只追加当前块，最后块不提交。首73帧及每个已发布前缀在 `.npy` 层逐像素不回改，15个视频/片段与 endpoint/cache 文件 SHA、FPS、帧数和单调PTS见 [完整性审计](artifacts/video_integrity.json)。

任务实际 **180 sampling + 6 clean commit = 186 denoiser forwards**，6次VAE，0训练更新，GPU 0 顺序运行两条路径，合计 **1572.27 GPU-seconds / 0.43674 GPU-hours**。记录到的新块 allocated 峰值 **26,876.70 MiB**；初始第三块 commit 只通过44GiB上限断言，精确峰值未保存。CPU KV 随历史增长至最后块采样前的 **18.131 GB**。逐块计时和与V2b的观察成本见 [效率表](efficiency_table.md) 与 [原始指标](metrics.json)；GPU transfer 是 sampling 的组成部分，不应重复相加。完整从零 E2E/首屏延迟未测，不能声明已经达到交互实时速度。

大 cache、latent endpoint 和未压缩RGB保留在外部 `H3-World/outputs/EXP-003_native_cached_124/`，通过 [来源与产物 manifest](MANIFEST.md)追溯，不复制进Git。正式任务书见 [taskbook_v1.md](taskbook_v1.md)；本轮已形成正式V3 Baseline可行性版本，保留上述视觉失败片段与效率范围。
