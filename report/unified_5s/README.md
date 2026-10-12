# V0–V4：统一 5 秒视频、生成耗时和峰值显存

**EXP-018 · 2026-10-13 · 七个配置均为本轮从噪声重新生成，零新增训练。**

[播放七版本同屏视频](videos/V0_V1_V2a_V2b_V3_V4_unified_5s.mp4) · [代表帧总览](all_versions_keyframes.jpg) · [CSV 表格](results.csv) · [完整 JSON](results.json)

**双栏原分辨率对照：** [左 V0 / 右 V3 Baseline](videos/V0_vs_v3_30_5s.mp4) · [左 V0 / 右 V3 FM-8NFE](videos/V0_vs_v3_8_5s.mp4)。均为本轮重新生成，两个面板各保留 832×480。

## 先说明长度和计时

相同停车场初图、提示词、持续 A、seed13、832×480、24fps、flow shift 2.22。H3 原生帧数为 `5+17k`，所以每行真实生成 **124 帧（5.167 秒）**，展示视频统一取前 **120 帧，严格 5.000 秒**。下表成本均对应完整 124 帧。各行独立加载模型、重新编码图文条件、重新生成首窗及后续块；没有借用历史首窗、端点或 KV。

## 统一成本表

同一 NVIDIA L40 / GPU5，统一 `ABOT_VRAM_RESERVE_GIB=18`，CPU offload。时间单位秒，内存单位 GiB。GPU 峰为 PyTorch allocated/reserved 的全流程峰（含准备、采样和解码），CPU KV 单列。

| 版本（点击单独播放 5 秒） | 推理采样设置 | 整片采样 NFE + KV 提交调用 | 完整生成耗时 s | GPU allocated 峰 GiB | GPU reserved 峰 GiB | CPU KV 峰 GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [V0 Original H3](videos/v0_5s.mp4) | 整片 30 NFE | 30 + 0 | ≈802.6 | 26.26 | 28.27 | 0.00 |
| [V1 Native causal](videos/v1_5s.mp4) | 8 块 × 8 NFE | 64 + 8 | 417.9 | 26.39 | 27.09 | 13.19 |
| [V2a RGB Anchor](videos/v2a_5s.mp4) | 8 块 × 8 NFE | 64 + 8 | 658.4 | 26.45 | 27.36 | 13.19 |
| [V2b Same-σ](videos/v2b_5s.mp4) | 6 块 × 30 NFE | 180 + 0 | 2172.0 | 26.26 | 27.38 | 0.00 |
| [V3 Baseline](videos/v3_30_5s.mp4) | 6 块 × 30 NFE | 180 + 5 | 1358.3 | 26.32 | 26.79 | 16.89 |
| [V3 FM-8NFE](videos/v3_8_5s.mp4) | 6 块 × 8 NFE | 48 + 5 | 505.1 | 26.32 | 26.87 | 16.89 |
| [V4 DMD cycle8](videos/v4_5s.mp4) | 6 块 × 8 NFE | 48 + 5 | 485.5 | 26.39 | 26.83 | 16.89 |

**计时限定：** 独立子进程 wall 包含 Python 导入、模型/适配器加载、条件编码、生成、解码、保存及退出；120 帧展示转码在计时之外。V0 导出后的帧率字符串断言导致外层 wall 未落盘，该行 `≈` 由启动记录到 124 帧 MP4 文件时间重建，截止到文件保存，不含进程退出；原推理完整成功，无 GPU 重跑。该来源差异保留在 JSON，不以小数精度掩盖。

**NFE 是推理网络求值次数。** V3 FM-8NFE 的 8 是每块 8 次，六块合计 48 次，加 5 次 clean KV commit；它不是 AnyFlow。V4 的 cycle8 表示历史 DMD 训练循环编号，与推理每块 8 NFE 是两个不同维度。本轮没有训练更新。

## 分阶段耗时与比较边界

| 版本 | 模型加载/准备 s | 本次条件编码 s | 采样循环 s | VAE 解码 s |
| --- | ---: | ---: | ---: | ---: |
| V0 Original H3 | 63.7 | 138.8 | 496.8 | 10.2 |
| V1 Native causal | 3.2 | 10.5 | 348.7 | 10.0 |
| V2a RGB Anchor | 3.7 | 10.5 | 588.6 | 10.9 |
| V2b Same-σ | 18.7 | 10.9 | 1985.1 | 44.9 |
| V3 Baseline | 18.8 | 10.4 | 1128.8 | 45.0 |
| V3 FM-8NFE | 19.7 | 10.8 | 266.3 | 39.1 |
| V4 DMD cycle8 | 19.2 | 10.5 | 267.7 | 39.5 |

这些阶段用于解释开销，不能简单相加当成统一总耗时：legacy 加载列只计模型/LoRA 加载，native 还计首轮 DiT 准备；逐块模型切换包含于 wall，但未全部进入这些子列。legacy 采样循环含其 KV commit/动态 anchor 处理，native 的 commit 另计；原生 V2b/V3/V4 每块解码可见前缀并保留已发布帧，legacy 最终整片解码。具体计时边界见原始指标和源码。主表 wall 保留上述所有实际成本。

**这是顺序执行、每配置一次的观测（n=1）。** 首条 V0 的模型加载和条件编码明显慢于后续条目，受磁盘/编译等缓存状态影响；不将总 wall 比值称为消除系统因素的纯算法加速。七行保留各自版本的精度、attention、anchor 和分块协议，是完整方案对比。V0/V1/V2a 使用历史 precision，native 行使用冻结 `h3_fp32` 配置（恢复原生边界投影/时间输入的 FP32，Transformer 主体加载 dtype 不变）。代码层面 V0 使用原始 FlexAttention 路径，分块 attention 路径使用 PyTorch SDPA；本轮没有额外 profiling 来确定 SDPA 内部最终选中的 CUDA kernel。V2a 含历史训练适配器，V4 含历史 DMD 权重，不能归因于单个 KV 改动。

## 实际画面与连续性

Judge 检查全部 124 帧接触表及必要原分辨率细节；执行完成与生成能力分开判断。本轮仅持续 A，没有 D 反事实或动作切换测试，也没有新增长视频验收。

| 版本 | 人物/场景结构 | 本轮观察 |
| --- | --- | --- |
| V0 Original H3 | USABLE | 人物保持单体完整，停车场整体可辨，横向运动连续；局部腿脚重影和纹理形变，没有持续人物解体。 |
| V1 Native causal | PARTIAL | 人物和场景尚可辨，后半段动作明显停滞，人物姿态接近冻结；亮度/对比度不断上升，地面和天花出现条纹、过锐边缘，不能作为动作能力通过。 |
| V2a RGB Anchor | USABLE_WITH_ARTIFACTS | 人物单体结构和停车场保持可辨，较V1没有明显持续过曝，但肢体拖影、纹理模糊仍在；运动偏向车库深处，与V0横向参考明显不同，不视为持续A控制已恢复。 |
| V2b Same-σ | USABLE_WITH_ARTIFACTS | 人物和场景保持完整可辨，横向运动明显；约56–73帧出现转身/姿态波动，腿脚和手臂短暂重影，块边界连续性仍为PARTIAL，没有持续人物解体。 |
| V3 Baseline | PARTIAL_TRANSIENT_DEFORMATION | 整体人物与车库可辨，但约80–91帧（3.3–3.8秒）出现明显人体折叠、肢体分离/短暂结构失真，之后恢复；108帧附近有额外透明残影，块边界及动作连续性为PARTIAL。 |
| V3 FM-8NFE | USABLE_WITH_ARTIFACTS | 人物与停车场持续可辨，本次未见V3-30NFE中段那样的明显身体折叠；腿部拖影和双边轮廓更持续，纹理模糊，后段天花/场景几何有倾斜漂移与边界变化。质量仍PARTIAL，不据单次视频宣称8NFE普遍优于30NFE。 |
| V4 DMD cycle8 | FAIL | 从第一帧到第124帧均为彩色噪声，无可辨人物或停车场；生成能力FAIL，不能从执行完成、finite数值或耗时较低推断有效。 |

## 这轮支持的判断

- **V3 FM-8NFE 是当前更值得优先使用的因果原型配置。** 本次完整耗时505.1s，相较同路线30NFE的1358.3s降低约62.8%；人物/运动可辨，但ghosting和场景漂移仍需保留PARTIAL结论。
- **V3 Baseline没有在本次完整wall上超过V0。** 它比V2b重算方案快，但当前1358.3s高于V0约802.6s；不同调用数、attention/精度、offload和系统缓存共同影响结果，不能把“因果+KV”直接等同于整片更快。
- **未测得明显GPU峰值显存下降。** 七行allocated峰约26.26–26.45GiB；V3额外占用CPU KV峰16.89GiB。统一offload限制下需同时考虑GPU与CPU成本。
- **V4当前DMD权重生成FAIL，停止该配置扩展。** 本轮提供真实完整负结果，不新增训练或挑选checkpoint。

七行累计614次采样+31次KV提交=645forward，0训练；含一次条件编码失败共6417.118s≈1.783单卡墙钟GPU小时。V0计时重建与n=1缓存差异的限制同上。[Judge最终审核](../../experiments/EXP-018_unified_5s_benchmark/judge/FINAL_REVIEW.md)。

## 版本目的

- **V0 Original**：原始整片双向 H3 + released Action LoRA，作为画面和成本参考。
- **V1 Native causal**：早期分块因果化，latent dual anchor，检查直接因果化后的退化。
- **V2a RGB Anchor**：RGB 双锚点和冻结 visual/action adapters，尝试稳定人物和场景。
- **V2b Same-σ**：每步把历史置于相同噪声水平并与当前双向重算，恢复结构和动作；没有 persistent KV。
- **V3 Baseline**：Single I0、strict causal、Global RoPE、persistent raw KV，每块 30 NFE。
- **V3 FM-8NFE**：同一 V3 路线的普通 Flow Matching 减步，衡量速度和质量代价。
- **V4 DMD cycle8**：Distribution Matching Distillation 历史最终学生权重的完整生成评估，属于 SGF 研究尝试；不是已验收正式生成模型。

本次 V3 只生成六块，未覆盖滑窗淘汰后的新块。V4 从第一块开始独立生成，较历史借用 FM 首窗的短续写评估范围更大；失败图像仍为真实模型输出，不补帧、不换 checkpoint。

## 证据与复现

[实验目录：日志、配置、124 帧原片、指标](../../experiments/EXP-018_unified_5s_benchmark/README.md) · [Judge 独立结果审计](../../experiments/EXP-018_unified_5s_benchmark/judge/results_audit.json) · [逐帧审核记录](../../experiments/EXP-018_unified_5s_benchmark/judge/quality_review.json) · [计时口径](../../experiments/EXP-018_unified_5s_benchmark/judge/MEASUREMENT_CLARIFICATION.md) · [展示视频来源](video_manifest.json) · [总览构建记录](presentation_manifest.json)

[报告首页](../README.md) · [历史 Original/V3 汇总](../original_vs_causal/README.md) · [术语解释](../GLOSSARY.md)
