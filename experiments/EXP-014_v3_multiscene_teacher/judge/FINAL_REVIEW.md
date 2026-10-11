# EXP-014/v1+v2 — Judge最终验收

2026-10-11 08:25 HKT。**ACCEPT WITH LIMITS：四训练初图的native V3 FM30教师目标生成任务完成，协议与来源PASS；四场景动作正确监督未成立。** 0训练，不启动基于这四组目标的AnyFlow/DMD，不加C3、换seed或调参。正式V3/FM8结果保持冻结。

## 实际证据与能力边界

Original H3+released Action LoRA、native Single I0、native30-step、Global RoPE、strict causal、own-action/current-prefix feedback和clean raw KV均沿用冻结V3。四图按EXP-013固定train候选选择。每图C1 A共39帧、同C1/cache/noise分叉AA/AD到56帧。合成A/D不等于原录屏动作GT。

Judge已检查全部292张唯一输出帧（4×39 C1+8×17续写）和必要原分辨率细节。四套实际50层C1 KV及native位置/动作/噪声、未来row裁剪、旧39RGB不变、完整解码/24fps核查通过。全帧观察见[visual_notes.json](visual_notes.json)。

| 固定训练初图 | AA/AD中央ROI flow累计 | 视觉与动作结论 | 允许的研究用途 |
| --- | --- | --- | --- |
| s0 山地树木 | +42.147 / −27.459 | 动作分叉/主体可用，软化等quality PARTIAL | 有限配对教师候选，单scene/seed限制 |
| s1 草地道路 | +3.654 / −45.314 | AD反转清楚；AA净位移弱且中途改变符号，持续A PARTIAL | 带明确限制的诊断目标，不能称无条件合格动作监督 |
| s2 工业道路 | +68.385 / +50.074 | 人物/场景可辨，两路主要同向，D反转未证实 | 不接受AD为动作正确监督，保留负结果 |
| s3 暗草地 | +66.178 / +58.664 | AA/AD接近，新增RGB MAD3.188；D反转未证实，暗处遮挡限制细节 | 不接受AD为动作正确监督，保留负结果 |

光流辅助配对视觉判断，不是准确率。没有持续全场景彩噪或人物大面积分解；普通画质问题保留PARTIAL。教师生成完成不证明训练会获益，也不把当前局部问题扩大为整个V3失败。

## 中断、恢复和真实成本

原T2 s1/s3在AD中断，原会话exit143/SIGTERM；没有Python异常，可见cgroup OOM计数为0，终止来源未知。原中断AD没有端点/视频，不作质量FAIL，账本未关闭的0秒不作零成本。

v2获独立授权后仅从保存的C1/cache/noise各重做缺失AD一次，顺序GPU0，两条均正常完成；没有重算C1/AA/commit。新增60sampling/2decode、318.691821GPU秒。原输出与中断账本未覆盖，独立协议审核见[恢复s1](RECOVERY_s1_7199292c_AUDIT.json)、[恢复s3](RECOVERY_s3_b784d995_AUDIT.json)。

累计**402次denoiser预记账/计费，至少400次确认完成**（其中4次clean commit）；原中断最后两次调用是否完成不确定。12decode完成，12text+4image encode，0backward/0training。教师GPU时间[2238.042493,2398.353151]秒，编码110.213939秒，总**[0.652293,0.696824] GPU小时**，预算内。详见[独立总账](FINAL_BUDGET_AUDIT.json)。

## 交付完整性和决策

原P1/T1/T2发布审计已通过；本次恢复新增20个副本逐字节核对原源/manifest，两个56帧1664×552比较片全解码与源画面对位PASS，MAD均值约1.89–2.45：[恢复发布审计](RECOVERY_PUBLICATION_AUDIT.json)。大tensor/KV留原路径，Git保留代码、日志、来源索引和视频。

本轮收口。下一项EXP-015准备同四训练episode的真实连续56帧及原始观测动作，先CPU验证来源、时间映射和native action转换；不能把录屏混合键标签改成纯A/D。真实C1 teacher forcing、合成teacher C1和student自己生成C1分别标注，后续训练另立任务预算。当前不追加四图AnyFlow训练，不挽救弱教师AD。
