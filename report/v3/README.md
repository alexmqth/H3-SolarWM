# V3：正式基线与两个Sliding Window候选

**V3 Original Feasibility Baseline继续作为已验收的正式基础版本。** V3-SW-G和V3-SW-L独立命名、独立配置、独立验收，候选结果不覆盖Baseline。此前短暂使用的V2c分类按用户最新决定撤回，历史迁移记录保留。

| 属性 | V3 Baseline | V3-SW-G | V3-SW-L |
| --- | --- | --- | --- |
| Backbone | Original H3 + released Action LoRA | 相同 | 相同 |
| 新训练 | 无 | 无 | 无 |
| Strict causal / Persistent raw KV | 是 / 是 | 保持 | 保持 |
| 历史窗口 | 已验收范围最多5个祖先 | 固定最近5个祖先 | 固定最近5个祖先 |
| 历史淘汰 | 124帧结束前未触发验证 | 已验收C7/C8真实淘汰 | 同左 |
| Video位置 | Global RoPE | Global RoPE | 显式Sliding Local RoPE |
| 超过6块 | 未验收 | 158帧有限可行性通过 | 158帧工程通过、生成PARTIAL |
| 定位 | 正式参考、124帧有限可行性 | 检查真实淘汰与KV容量 | 检查位置外推和适配风险 |

- [V3 Baseline：124帧正式证据](v3_baseline/README.md)
- [V3-SW-G：Sliding Window + Global RoPE](v3_sw_g/README.md)
- [V3-SW-L：Sliding Window + Local RoPE](v3_sw_l/README.md)

EXP-005/v2已完成281forward/9VAE，实际0.662707GPU小时。SW-G优先保留；SW-L有动作响应但重复场景/亮度跳变更多，无已证实收益，当前无训练方向归档。正式Baseline不变。普通FM减步与AnyFlow finite-map训练已分别完成有限验证，见下方各自证据。


## 全程普通FM8

[EXP-006 V3-FM8](v3_fm8/README.md)已验收：新8步首39+AA/AD73，有限可行性通过、quality PARTIAL。43forward/5VAE/0.136905GPU小时，零新训练。普通减步与finite-map训练分别记录；EXP-010已进一步验收全FM8+SW-G各158帧，画质与首晚切换块响应PARTIAL。

## AnyFlow研究进展

[V3-AF](v3_anyflow/README.md)：累计32次finite-map训练和8NFE AA/AD73续写有限可行性通过，quality PARTIAL；共同FM8首窗，尚无整体优于FM8的证据。正式V3 Baseline冻结；EXP-009的4NFE评估已完成，AF4 AD第三块失败，当前配置归档。

## DMD研究进展

[V3-DMD](v3_dmd/README.md)：真实33B单cycle完整8map反向工程通过，17forward/3backward/3update/0VAE，三卡0.212416GPUh；累计8cycle训练工程通过，但最终AA C2全17新增帧彩噪，生成FAIL、当前配置归档；AD中断未评估，C3未执行。整个EXP-008约1.66409GPUh。

## 4NFE有限对照

[V3 FM4 / AF4](v3_fm4_af4/README.md)：EXP-009完成共同FM8首39后的AA/AD73。普通FM4有限可行PASS、quality PARTIAL；AF4 AD C3持续人物结构失败，生成FAIL、当前配置归档。38forward/8VAE/.217303GPUh，零新训练，不扫2NFE。EXP-010全FM8+SW-G158已验收；EXP-011已验收两个固定其他场景FM30/FM8的AA/AD56短程可行性，quality PARTIAL。

## 固定其他场景的短程迁移

[EXP-011 工业/村落FM30与FM8对照](v3_scene_transfer/README.md)：四配置AA/AD56均有限可行PASS，村落FM8颗粒/重影更明显；0训练/0.514521GPUh。只覆盖固定两初图与短程，不作广泛泛化结论。

## 四训练初图的原生FM30教师候选

[EXP-014 四图AA/AD 56帧画廊](v3_teacher_targets/README.md)提供独立可播放的四条并排视频和逐场用途标签。四图的C1/C2协议与真实raw KV均完成；`s0`有限可用，`s1`的D反向清楚但持续A弱，`s2/s3`没有证明D反向。两条原AD中断由独立恢复入口仅补缺失AD；四图不能直接当作全合格动作监督，也不代表AnyFlow或DMD已训练。

## 冻结AF8迁移负结果

[EXP-012](v3_anyflow/README.md)：共同FM8首窗、匹配8NFE，工业D切换PARTIAL，村落AA后半C2结构FAIL；协议/真实cache通过但无联合迁移收益。34forward/4decode/.098445GPUh，零训练。当前checkpoint停止扩展，普通FM8优先。
