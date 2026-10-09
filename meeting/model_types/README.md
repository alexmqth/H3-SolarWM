# 三类模型／推理配置：展示视频与训练来源

**按“是否新增训练、训练目标是什么”分成三类。** 这是模型／配置的分类；“单窗口、固定GT/reference history、自由rollout”是另一条评测维度。不能用后者代替前者，也不能把所有8步推理都叫AnyFlow。

本目录集中保存四条可直接播放的完整对比片。01及两条03视频为原片逐字节复制；02是把既有Original/FM0/FM48单条结果按帧并排，未重新运行模型。全部39帧、24fps，约1.625秒，保留完整失败后段，无补帧、循环、平滑或变速。124帧会议对比另有链接。

## 分类定义

| 类型 | 是否新增训练 | 发生了什么变化 | 研究用途 | 代表版本 |
|---|---|---|---|---|
| **I：Original H3＋causal routing** | **否，0 updates**；相对于已发布H3-World | 使用原始权重，改变可见窗口、路由、历史输入或缓存规则 | 检查直接因果化对画面与动作的影响 | FM实验的`step_00`；零训练T1/T2、C/N历史条件诊断 |
| **II：Original H3＋causal adaptation** | **是，ordinary FM、visual/action adapter等** | 对因果执行方式训练适配参数，不使用AnyFlow有限区间目标 | 检查训练是否恢复动作、结构和历史衔接 | 真实ABot FM48、fixed-mix、RGB visual、E2；匹配的FM16 control |
| **III：Causal H3＋AnyFlow** | **是，TF-AnyFlow finite-map训练** | 在因果初始化上训练有限时间区间映射，包含diagonal/diffusion与有限区间样本 | 检查4/8步能否保留生成效果 | AnyFlow16、full-history系列、128/136对照 |

Original H3完整双向推理是三类之外的共同**参考模型**。发布的H3-World action LoRA属于底座，不算本项目新增的causal adaptation。DMD/Stage2属于另一训练目标，不混入这三类。

“类型III”描述使用了AnyFlow目标，不意味着作为起点的causal模型已通过质量验收，也不意味着最终few-step成功。我们实际使用的causal初始化仍有质量局限。

## 01：直接因果化——Original vs 零更新causal

**[播放 01_routing_only_39f.mp4](01_routing_only_39f.mp4)**

| 视频位置 | 模型／配置 | 采样 | 评测条件 |
|---|---|---|---|
| 左列 | Original H3参考 | 30整段steps | 完整39帧双向推理 |
| 中列 | **类型I：Causal0，新增训练0次** | 30steps/chunk | 自生成history、5＋5＋2latent chunks、CPU raw KV |
| 右列 | **类型I：同一个Causal0** | 8steps/chunk | 同一因果配置；减少solver步数，没有AnyFlow |

上排A，下排D。中列人物仍在，但A/D运动相似；右列约20–22帧后明显重影。30步Causal0的A/D flow为−0.1955/−0.1870，Original参考为+1.1813/−0.8421。**没有新增训练时就已出现动作失配。**

这里的Original与causal在precision、anchor/prefix等推理处理上不同，属于整个推理协议对比，不能只归因于一张attention mask。`step_00`加载了零输出的adapter容器，旧运行字段`trained_for_causal=true`仅反映加载路径；实际更新数为0、没有继承已训练visual初始化，见[初始化审计](../../reports/stage1_anyflow/01_real_video/real_abot_fm/initialization_audit.json)。分类依据是实际参数和训练记录，不是字段名。

[源视频及完整评审](../../reports/stage1_anyflow/01_real_video/real_abot_fm/report/parking_baseline_complete/README.md)。

## 02：因果适配——同一因果配置的FM0 vs FM48

**[播放 02_adaptation_fm0_vs_fm48_39f.mp4](02_adaptation_fm0_vs_fm48_39f.mp4)** · [第31帧预览](02_preview_frame30.jpg)

| 视频位置 | 模型／配置 | 新增训练 | 采样 |
|---|---|---:|---|
| 左列 | Original H3参考 | 0 | 30整段steps |
| 中列 | **类型I：Causal0** | 0 | 30steps/chunk |
| 右列 | **类型II：Causal FM48** | 普通FM，48 updates | 30steps/chunk |

上排A，下排D。中、右保持相同因果推理协议、精度、首帧、prompt、动作、seed/noise；都使用自己的generated history及CPU raw KV，具体生成的历史会随参数改变。右列从Original＋released action LoRA的零适配初始化出发，用真实ABot数据集16 train / 8 validation训练rank8 QKVO/FFN bank，训练保留GT clean-history梯度；未使用AnyFlow或DMD。

| 30步/块 | A flow | D flow | 观察 |
|---|---:|---:|---|
| FM0 | −0.195516 | −0.186950 | 人物/车库可辨，动作近同向 |
| FM48 | −0.182680 | −0.163995 | 未恢复正确A方向，未显示明确质变 |

**这条片直接展示“做了causal adaptation，但本轮训练未恢复控制”。** 不能把完成48次更新写成恢复成功。Original参考仍保留legacy精度差异，中/右两列才是匹配的训练前后比较。[FM48完整结果](../../reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_COMPLETE_REVIEW.md) · [训练记录](../../reports/stage1_anyflow/01_real_video/real_abot_fm/training_snapshot.json)。

类型II还有两个重要展示版本：

- **[124帧 Original／旧fixed-mix／RGB visual](../action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)**：中、右都属于类型II，都没有AnyFlow。旧版动作符号相对较好但后段漂移；RGB版结构相对完整但A方向错误。它们同时改变多项配置，不是单因素训练对照。
- **[真实数据视频的GT／Original／FM0／FM48](../../reports/stage1_anyflow/01_real_video/real_abot_fm/report/trained_complete_generated30/dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4)**：中世纪场景generated-history约23帧后出现残影，后段人物分解；保留实际负结果。

## 03：少步训练——匹配的FM16 vs AnyFlow16

**[播放 03_anyflow_vs_fm_4step_39f.mp4](03_anyflow_vs_fm_4step_39f.mp4)** · **[播放 03_anyflow_vs_fm_8step_39f.mp4](03_anyflow_vs_fm_8step_39f.mp4)**

| 视频位置 | 模型／配置 | 新增训练目标 | 推理 |
|---|---|---|---|
| 左列 | Original H3参考 | 无 | 30整段steps |
| 中列 | **类型II：FM16** | 普通FM，新增16 updates | 对应文件的4或8steps/chunk |
| 右列 | **类型III：AnyFlow16** | TF-AnyFlow，新增16 updates | 对应文件的4或8steps/chunk |

上排A，下排D。两组从**同一个已训练RGB causal初始化**出发，共享visual/action初始化、rank8 bank容量、两条Original H3 A/D伪标签、LR、seed、更新次数与因果协议；两者都detach历史KV。visual/action适配已经存在，16指本轮新增更新，不是这些权重一生只训练过16次。FM48真实数据实验不是这组AnyFlow的父checkpoint。

FM不安装目标时间MLP；AnyFlow增加冻结的目标时间条件，有限区间目标需要额外前向。因此这是匹配初始化/容量/数据/更新数的目标和采样对照，**不是等计算量对照，也不是仅替换一个loss标量**。

| 配置 | A flow | D flow | 完整39帧观察 |
|---|---:|---:|---|
| FM16，4步/块 | −1.0764 | −1.2901 | 比AnyFlow4保留更多结构，仍有透明/模糊 |
| AnyFlow16，4步/块 | −1.1985 | −1.2421 | 后段更明显雾化/重影 |
| FM16，8步/块 | −1.1353 | −1.4562 | 主体可辨，A方向错误 |
| AnyFlow16，8步/块 | −1.3053 | −1.4965 | 无明确优势，A方向错误 |

结论限于这次两条伪标签、16新增更新的小试，不能推广为AnyFlow无效。[匹配实验完整报告](../../reports/stage1_anyflow/03_anyflow_trials/fullscope_fm_control/FINAL_RESULTS.md)。更后面的[128／control136／auxiliary136，8步完整对比](../../reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/original128_control_auxiliary136_8step_AD.mp4)也属于类型III，但仍未通过画质/action gate；这里优先用FM16/AnyFlow16展示是因为两者的训练比较更清楚。

## 模型类型与评测条件是两个独立维度

| 已有结果 | 模型／配置类型 | 评测条件 |
|---|---|---|
| 单窗口39f、原生时间＋单I0的好视频 | Original正控，不是多块causal通过 | 无历史、整个当前窗口动作已知 |
| 零训练T2/N历史条件视频 | 类型I | 固定Original参考history、局部续写；逐sigma重算，无persistent hidden KV |
| E2 FM-only4 / FM＋action4 | 类型II | 固定reference或真实GT-history局部续写；无persistent hidden KV |
| fixed-mix / RGB124f主片 | 类型II | 自生成history，自由rollout，persistent CPU KV |
| 本页AnyFlow16及后续128/136 | 类型III | 所链接完整片使用自生成history、persistent CPU KV |

因而“Original＋causal routing”既可能用于固定历史诊断，也可能用于自由rollout；“有history”不等于“训练过”，“8步”不等于“AnyFlow”，“有KV”不等于“画面稳定”。[原有因果基线评测说明](../../docs/CAUSAL_BASELINE.md)继续保留，各视频不能跨协议直接排成学习曲线。

## 播放与复核

现场按01→02→03的顺序，分别讲：**直接因果化改变了什么 → 普通训练是否恢复 → AnyFlow少步是否有效**。三个问题均有完整负结果，不把这次归类说成三个已成功完成的训练阶段。

所有39f片的causal是3块：30步/块＝90 noisy forwards＋3 clean commits；8步/块＝24＋3；4步/块＝12＋3。Original30是30整段去噪。片上时间是原始共享硬件单次记录，不是统一硬件warmup均值，不能据此声称加速。视频MAD与水平光流也不是完整画质或动作准确率。

[逐文件来源、分类与解码记录](manifest.json)。02的[拼接配置](02_adaptation_30step_spec.json)可在提交包根目录重建：

```bash
python meeting/render_comparison.py \
  --spec meeting/model_types/02_adaptation_30step_spec.json \
  --source-root . \
  --output meeting/model_types/02_adaptation_fm0_vs_fm48_39f.mp4
```

渲染只需PyAV、NumPy、Pillow和已有视频，不需要模型权重/GPU。Original两条源视频已放在`sources/`，causal源视频引用包内归档。FM48/AnyFlow的新权重仍在本机原outputs；本目录发布的是可播放结果与可核对配置，不声称附带所有新checkpoint张量。本次没有追加训练、模型推理或改变任何实验结论。
