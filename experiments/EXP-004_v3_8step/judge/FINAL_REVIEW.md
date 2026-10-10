# EXP-004 / v1：Judge 最终验收

2026-10-10，Asia/Hong_Kong。Decision: **accept**；Worker execution: **completed**；Judge acceptance: **accepted**。按当前可行性阶段与用户ROI要求，证据足以回答本轮问题，实验收口。

## 结论

V3原权重普通native FM/Euler的新块采样从30步减为8步后，AA/AD在共同30步首窗上保留可辨的动作响应，且各自接入自己的8步第二块后均续到73 RGB帧（24fps，3.04秒），人物/停车场基本可用。协议有效、严格因果与真实KV复用维持，零新增训练。

这支持**8步续写有限可行性**。首12 latent/39 RGB来自既有30步流程；全程8步初始化、8步124帧、首屏/完整E2E与跨scene/seed未测。普通FM减步证据不能称AnyFlow训练或DMD成功。正式30步V3的124帧定义不变。

| 能力/执行项 | 判定 | 范围 |
| --- | --- | --- |
| 执行、来源与预算 | PASS | 4条8步新块，2次own-history commit，实际资源闭合 |
| strict causal / persistent KV | PASS | 继承已验收图与路由，采样不改历史cache，无额外诊断 |
| 动作响应 | PASS（有限） | 第二块同history AA/AD对照；第三块仅闭环续写 |
| 基本人/场景结构 | PASS（有限） | 全部新增68帧检查，2条至73帧 |
| 画质、严格连续性 | PARTIAL | AA后段肢体透明拖影明显；软化、纹理重叠、姿态跳变 |
| 新块采样减步与观察成本 | PASS | 实际8步；旧30步计时作观察参考 |
| 全程8步、长片与公平E2E | NOT_TESTED | 不将借用30步首窗和不同时刻计时外推 |

## 数值与视觉

| 路径/新RGB | 8步flow | 旧30步flow | 8步sampling秒 | 旧30步sampling秒 | 边界灰度MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| AA/39–55 | +0.780621 | +1.347308 | 70.869 | 142.913 | 3.539 |
| AD/39–55 | −1.509824 | −1.457716 | 40.524 | 135.687 | 2.814 |
| AA/56–72 | +0.978349 | +0.473785 | 37.764 | 155.140 | 4.895 |
| AD/56–72 | −0.732276 | −0.929845 | 42.510 | 146.390 | 7.953 |

Judge看过四块的完整新增帧静态序列及原分辨率末帧。第二块AA/AD人物场景基本可辨，有软化/背景纹理重叠；AA第三块后段透明肢体、腿部拖影/重复更明显，AD第三块局部腿部纹理与残影。未见持续整体人体崩坏，不能描述为无缺陷。flow只是运动代理，不单独作为动作语义评分。第三块历史已分化，方向差异不能称同状态反事实。

## 核验与成本

- Runner复用EXP-002 interval_cached与current-prefix；native 8步schedule含9个sigma端点、shift2.22不变。第二块不commit；第三块前只提交自己的8步第二块一次，末第三块不commit。
- 独立比对第二块AA/AD共同history/noise/cache/anchor/audio/global位置/旧RGB，当前动作条件不同；每条与旧30步输入条件匹配。第三块endpoint来源确属自己8步输出。
- 旧RGB前缀逐像素冻结、endpoint/RGB摘要、采样前后内存cache identity/storage/version与文件摘要、权重版本检查通过。Runner/config/interval/router当前hash与4个运行记录全部一致，详见[成本/来源核对](costs_checked.json)。
- 四个原片完整解码、24fps、帧数/PTS通过；四个30vs8并排片来源/输出hash、尺寸1664×560、帧数、24fps/PTS通过。检查了实际并排帧及文字：左30右8，RGB56起明确各自历史。见[comparison_checks.json](comparison_checks.json)。
- 32sampling+2commit=**34 denoiser forwards**，**4VAE**，0训练/额外模型诊断/首窗prefill/reencode。GPU0四个顺序进程均完成，**466.01794 GPU秒=0.12944943 GPU小时**；首进程至末进程结束573.55258秒，**峰值allocated25,682.066MiB**，低于34forward/4VAE/0.5GPU小时/90分钟/44GiB预算。
- 历史raw KV由首12的6,799,104,000 bytes增至through17的9,632,064,000 bytes，最后第三块不提交。sampling含cache读入/搬运，子项未拆分。旧30步为不同时刻共享硬件单次运行，不声称精确或公平E2E速度比；首窗原始成本未包含。

Worker建议将整体局部能力标为PARTIAL；Judge保留其原文，独立区分基本可行性PASS与画质/连续性PARTIAL，未更改Worker原判断。最终报告、manifest与CPU核验日志已审阅，13份小型source文件及29份产物hash匹配；大型LoRA/KV复用实际运行摘要与先前验证，不增加模型诊断。

## 决策与交付

接受本轮有限正证据，停止GPU与额外调参。保留Worker原报告/指标快照和原始日志，独立Judge结论不回写伪造旧报告。8步结果作为V3增量效率证据进入submission/report，30步V3正式基线继续保留。

下一项最有价值的问题：让首窗也使用8步，检查全程少步能否在自己的历史下成立，再决定是否有必要训练。建议短窗可用再有限延伸；不扫描步数/shift/gain，不为局部画质补大量实验。EXP-004预算已关闭，建议不构成EXP-005执行授权。

[实验说明](../README.md) · [原Worker报告](../worker_report.md) · [原Worker指标](../worker_metrics_snapshot.json) · [任务书](../taskbook_v1.md) · [在途逐块审核](S0_REVIEW.md)
