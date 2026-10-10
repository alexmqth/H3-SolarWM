# EXP-012/v1 Worker 最终报告：冻结 AF8 的匹配历史场景迁移

2026-10-11 07:06 HKT。P0 与两次单独放行的 GPU 场景均已完成，未训练、未调参、未扩展 C3。[Judge 最终验收](judge/FINAL_REVIEW.md)判定**工程协议 PASS；匹配8NFE下没有稳定的动作与画质联合收益。** 工业场景 AF8 的 AD 反向动作响应明显弱于普通 FM8；村落场景虽保留 A/D 方向分叉，但 AA 后半块人物与前景出现持续的大幅透明条纹，视觉结构失败。本报告陈述 Worker 实际执行与证据。

## 固定比较与执行

唯一 student 为 EXP-007 AF2 step32 的配套 QKV rank8 blocks42–49 与 target-time gate0.25；来源、元数据和 SHA 见 [P0 报告](P0_REPORT.md)。两个场景各自复用 EXP-011 已验收的普通 FM8 **同一个首12 latent / 前39 RGB**，不生成新的首窗，也不读取普通 FM8 hidden KV。AF student 用自身权重、`sigma=target_sigma=0` 重新提交 C1 的50层 CPU raw KV；AA/AD 再共享这个 AF cache 与 C2 初噪声，仅切换当前 action spans。8 个 native sigma 邻接 finite-map 调用生成5 latent/17新RGB，到56帧。普通 FM8 对照直接引用 EXP-011 已保存结果，所以这是 matched-history model comparison，**两模型权重及自建 KV 不同**。

工业、村落均在 GPU0 按 [各自 marker](judge/) 顺序运行一次，环境与冻结命令见 [配置](config.json)和 [runner](run_exp012.py)。实际 **32 sampling + 2 clean commit =34 denoiser forward、4 VAE decode、0 text/image encode、0 backward/update、354.403 GPU 秒（0.09845 GPUh）**；低于1260秒上限，最高 allocated 26.230 GiB。两份 AF 自身 KV 各为6,799,104,000 bytes（约6.33 GiB），原始大缓存/latent 留在 `H3-World/outputs/EXP-012_v3_af8_scene_transfer/`。真实逐调用 [最终账本](artifacts/budget_final.json) 与两条原始日志已复制归档。

| 场景 | GPU wall s | Load model s | Commit s | Cache save/hash s | AA/AD sample s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| 工业 | 164.863 | 6.450 | 16.448 | 10.135 | 38.323 / 41.925 | 26.223 |
| 村落 | 180.996 | 6.143 | 15.148 | 10.116 | 37.711 / 49.237 | 26.230 |

这些是独立进程阶段计时，含模型加载、cache 保存、解码和文件写入；不能把8 NFE或 sampling 秒数直接等同产品端到端延迟。CPU KV 存储量也必须计入系统成本。

## 协议审计与质量结果

[Worker CPU 审计](artifacts/audits/)和 Judge 独立核查确认：配对 checkpoint 是同一个step32；每场景 AF-own cache 实际包含50层 index0，且不同于 FM8 cache；AA/AD 的 C1、C2 初噪声、native8 sigma、Global video position及非动作条件相同，动作差异只在当前 C2 action rows；推理没有改写 cache；所有视频的旧39 RGB与共同 FM8 首窗**逐值一致**。四条56帧和四条新增17帧 H.264 MP4 均为832×480、24 FPS、PTS唯一且可完整解码。这些证明执行协议成立，不能代替画质验收。

| 场景/动作 | 普通 FM8 C2 水平流 | AF8 C2 水平流 | 普通 FM8 边界 MAD | AF8 边界 MAD | 视觉/控制判断 |
|---|---:|---:|---:|---:|---|
| 工业 AA | +28.20 | +28.14 | 11.90 | 12.46 | 人物与工业场景仍可辨，A持续响应相近 |
| 工业 AD | −38.27 | **+11.07** | 11.02 | 13.16 | AF8切D后反向响应不清楚；不能判动作通过 |
| 村落 AA | +87.82 | +70.33 | 10.70 | 12.40 | 运动方向保留，但后半块人物/树叶透明条纹持续，结构FAIL |
| 村落 AD | −44.23 | −68.63 | 12.92 | **18.19** | 反向运动可辨，人物/村落仍可用；边界跳变更大 |

水平流按中央ROI的 Farneback 帧间中位水平位移，在C2的16次转移上求和；仅辅助看视频中的方向分叉，不是动作准确率标签，也不能跨场景按绝对值排名。MAD为原始发布RGB的帧38→39平均绝对差。村落 AA 的段内MAD甚至比FM8略低，但[末帧并排图](artifacts/comparisons/village_AA_last_pair.jpg)清楚显示它是人物与前景叠成条纹，不能以平滑指标误判为画质改善。工业 AD 的[末帧并排图](artifacts/comparisons/industrial_AD_last_pair.jpg)展示两方法的运动路径差别。Judge 已目视全部68张新帧与原分辨率代表帧；其评价应作为正式质量结论。

## 视频与原始证据

下列四条均为左普通 FM8、右 AF2 step32，前39帧使用相同 C1 RGB，从帧39开始比较 C2：

| 场景 | AA 持续A | AD 切D |
|---|---|---|
| 工业 | [FM8 vs AF8 AA](artifacts/comparisons/industrial_AA_FM8_vs_AF8_56.mp4) | [FM8 vs AF8 AD](artifacts/comparisons/industrial_AD_FM8_vs_AF8_56.mp4) |
| 村落 | [FM8 vs AF8 AA](artifacts/comparisons/village_AA_FM8_vs_AF8_56.mp4) | [FM8 vs AF8 AD](artifacts/comparisons/village_AD_FM8_vs_AF8_56.mp4) |

四条 AF 原片、四段新17帧、全帧图、原结果 JSON、原始运行日志、辅助审计与预算副本在 [artifacts](artifacts/)；逐文件 SHA、原始本机位置和文件类别见 [artifact manifest](artifacts/manifest.json)。工业首次对照与村落结构失败都保留，没有筛掉负结果。

## 判断与边界

本轮没有证据表明当前**有限停车场训练的 AF2 step32**能在两张新初图上稳定优于普通 FM8：工业 AD 动作退化，村落 AA 视觉结构失败，且每个场景只有一个固定初图、一个seed、一个续写块。该负结果不否定 AnyFlow 方法或已验收的停车场 AF8 局部可行性，只限制**这个 checkpoint 在这套匹配历史协议下的迁移结论**。按任务书在56帧收口，不新增训练、权重选择、C3、DMD或场景筛选；后续若研究多场景数据/训练，应另立任务和预算。
