# EXP-011/v1 Judge 最终验收

2026-10-11 06:42 HKT。**ACCEPT：两个固定ABot场景中，V3 causal FM30与全程普通FM8的首39帧及各自AA/AD56帧，有限可行性通过；四配置画质均PARTIAL。** 不加C3、不调参、不加场景。正式30-step V3 Baseline保持冻结。

## 实际执行和独立证据

输入来自工业与村落两个既有validation episode的原始PNG与静态场景描述，重新编码native Single I0/full37 fixture；未读取旧Dual Anchor编码。保持Original H3/released Action LoRA、native timestep、own-action/action feedback/current-prefix、Global RoPE、clean persistent CPU raw KV、seed13、12+5 latent；没有新增训练。

P1实际6 text encoder、2 image VAE encode，含两次模型调用前失败共70.564698 GPU秒。G1四首窗76sampling/4decode；G2四配置152sampling+4commit/8decode；总 **232 denoiser forward、12 decode、0 update，推理1781.709743 GPU秒，含P1共1852.274441秒=0.514520678 GPUh**。推理allocated峰26.119292GiB，P1编码峰40.576426GiB；所有GPU阶段完成且未超预算。见[最终账本审计](FINAL_BUDGET_AUDIT.json)与[冻结账本](budget_final_frozen.json)。

独立审核包括：原始PNG/静态描述、真实Single I0/noise/位置和动作行；8套G1/G2阶段来源/端点/native sigma/帧数/24fps/递增PTS；实际加载四套C1 cache，全部50层精确index0，每套6,799,104,000 bytes；每个方法AA/AD共享自己的C1与C2噪声，只有动作prompt行改变；所有G2原39 RGB逐值不变。逐配置审计保存在本目录G1/G2_*_audit.json。只有一个历史块，本轮不构成新的滑窗淘汰证据。

## 视觉与动作

Judge看完G1全部156帧和G2全部136新增帧，并检查原始初图、遮挡与末帧原分辨率。见[逐帧观察](visual_notes.json)。

| 场景/方法 | C2 AA辅助flow | C2 AD辅助flow | 实际判断 |
| --- | ---: | ---: | --- |
| 工业FM30 | +26.7002 | −43.9849 | 人物/桥梁/建筑保留，运动分叉可辨；细节软、边界视角调整 |
| 工业FM8 | +28.2013 | −38.2703 | 人物和场景可用，切换可辨；细节软、植被纹理有颗粒 |
| 村落FM30 | +68.0941 | −94.4928 | 相反场景运动清楚，人物短暂树后遮挡后重现；暗部/出框/软细节 |
| 村落FM8 | +87.8241 | −44.2290 | 人物/村落和运动分叉仍可辨；AA树叶、人体边缘颗粒和重影明显较重 |

这里flow为**中央ROI每次帧间中值水平流在C2的16次转移上求和**，与旧停车场均值指标不同；仅辅助成对视频观察，不是控制准确率，也非绝对值越大越好。树木遮挡不当作持续人体消失。没有持续全画面彩噪或完整场景崩坏；普通视觉缺陷按可行性尺度接受。

## 少步收益与比较边界

G1首窗没有生成历史，输入/噪声/权重/协议匹配，仅步数不同：工业FM30/FM8采样171.237/51.874秒，村落195.283/52.422秒，采样部分约3.30×/3.73×。两次观测不等于稳定性能benchmark，加载/编码/解码/提交需另计。

G2两方法分别使用自己生成的C1及KV，属于闭环方案比较，不能称相同raw KV的步数消融。AA/AD则在同方法内共享C1和噪声。FM30是causal V3，不是Original双向H3。每场景只有固定初图/seed13、56帧约2.33秒；不是统计泛化、多场景长时稳定或绝对未见场景证明。村落FM8质量代价真实存在，不以可行PASS抹掉。

## 决策

普通FM8在两固定非停车场初图上仍有有限短程动作/结构可行性，值得保留低成本候选；正式参考仍为FM30。此轮到56帧收口。下一项可用冻结AF2 step32、复用本轮FM8共同C1，检查匹配8NFE的短程迁移；必须独立任务/预算/CPU审核，不追加训练或挽救已归档AF4/DMD。设计见[候选草案](AF8_TRANSFER_DESIGN_DRAFT.md)。

交付视频及原始日志由artifacts/G2和Worker最终报告索引；发布前另核对归档副本、比较片与manifest，不更改原始计算状态来伪造通过。
