# 当前：V3-AF初始化与单步训练

EXP-006全程普通FM8已经验收：新8步首39+AA/AD73，有限可行性通过、quality PARTIAL，零训练。普通8步不再依赖30步首窗；长时全程8步仍未测。

EXP-007 AF0实际tiny-H3 CPU核查已通过。当前已批准新的target-time student + last8 rank8 QKV初始化及一次finite-map update，最多22forward/4backward/1update/0VAE/0.75GPU小时。保持V3 native/current-prefix/Global/strict causal/cache条件；更新后按student权重刷新KV。后续有限训练与匹配8NFE评估按实际成本另批，不因loss下降直接验收。

用户授权Judge夜间持续推进至2026-10-11 09:00，期间可用所有实际空闲GPU，按任务预算执行，不抢占他人进程；09:00后最多3卡。DMD需真实student samples、独立fake-score、正确角色/梯度与明确预算，不能沿用旧单次replay结果冒充V3多步训练。

[FM8验收](../../experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [AF CPU核查](../../experiments/EXP-007_v3_anyflow/AF0_REVIEW.md) · [当前AF任务书](../../experiments/EXP-007_v3_anyflow/taskbook_v2.md) · [DMD准备](../../experiments/EXP-007_v3_anyflow/FUTURE_DMD.md)
