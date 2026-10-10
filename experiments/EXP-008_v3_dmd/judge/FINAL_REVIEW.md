# EXP-008 最终Judge结论：工程通过，当前DMD配置生成失败并归档

2026-10-11。**stop/archive：cycle8的AA C2生成能力FAIL。** 新增17帧全部为彩色噪声，人物与停车场持续不可辨；这不是可接受的普通边界或残影缺陷。停止当前DMD配置，不增加训练、不换中间checkpoint挑片、不继续C3。正式V3 Baseline、FM8和AF8结果保持。

## 真实已测范围

v1真实33B单cycle、v2新增7cycles的完整8map梯度链、角色隔离、自身KV与checkpoint恢复工程均已通过，合计student8次、fake9次更新。最终仅cycle8用于视频评估，不从中间结果选择。

评估成功完成共享FM8 C1 prefill与AA C2，到56帧，其中前39帧为共同FM8且逐值不变，新增17帧全部噪声。Judge看过全部新增帧及与FM8/AF8的并排代表帧。两项基线在相同C1、动作、初噪声、native8step和V3条件下仍有可辨人物/场景。

AD C2在Judge停止指令发出时已启动；第一次“停止剩余评估”与随后“允许在途AD结束”的指令交叉，Worker已发送SIGINT。AD完成5个采样步，第6个forward期间中断，没有完整端点或视频。**AD为INTERRUPTED/NOT_EVALUATED，不记为方向失败或完成；AA/AD C3均未执行。** 不重跑AD补数量。

## 独立审计与诊断

[部分评估审计](eval_c2_stop_audit.json)通过来源/执行核查：21项冻结来源、最终cycle8、AA C2与FM8匹配的noise/prompt/sigma/clean历史、raw KV来源与50层indices、保存endpoint/RGB/video SHA、旧39RGB逐值不变、56帧24fps完整解码与递增PTS；另外核对AD中断状态及C3不存在。审计通过仅说明负结果可追溯，不代表生成通过。

[CPU端点诊断](aa_endpoint_diagnostic.json)：DMD AA C2与初噪声cosine为0.94939，FM8为0.09987、AF8为0.11365；DMD端点仍finite且RMS约1.044。支持“当前最终模型没有有效去噪”的观察，不能单独归因为学习率、critic、normalizer或某一个模块。训练中fake loss尖峰及后续梯度缩小记录保留，surrogate下降未带来可用视频。

## 实际成本

| 阶段 | 显式forward | backward / update | VAE | GPU小时 |
| --- | ---: | ---: | ---: | ---: |
| v1 pilot | 17 | 3 / 3 | 0 | 0.21241552（三卡保守） |
| v2延续训练 | 99 | 14 / 14 | 0 | 1.39470081（三卡保守） |
| 最终评估及中断 | 16已启动或预记账 | 0 / 0 | 1 | 0.05697321（205.103548秒） |

评估16包含14 sampling预记账，其中13完整完成、1中断，以及2次完成的clean commit。不能把16都称完整成功forward。成功阶段allocated峰值25.96971GiB，中断AD没有完整终态peak记录。整个DMD任务约1.66409GPU小时；没有C3消耗或追加训练。

## 研究决策

当前固定C1/current-block on-policy、有限fake更新和现有优化配置归档为负结果，不把它推广为所有DMD方法不可行。它也不是完整SolarWM Stage2或原生双向teacher设置。短预算阶段没有理由为此配置继续扫学习率、critic比例或更多cycles。

下一有价值的问题是现有可用AF2 student能否在4NFE保持比普通FM4更好的动作/结构，用固定checkpoint、同输入的有限推理比较回答，无新增训练；这直接检验AnyFlow减少采样调用的作用。另立EXP-009任务和预算，不复用本DMD授权。
