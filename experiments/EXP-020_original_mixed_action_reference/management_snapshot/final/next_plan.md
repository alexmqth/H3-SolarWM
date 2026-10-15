# 当前任务与Judge决定

2026-10-15 HKT。**EXP-019/020已全部执行完成并accepted收口；没有遗留GPU任务。**

## 已结束任务验收

- EXP-019：固定真实动作布局数值/来源通过，四scene GT39帧历史下FM-30NFE/FM-8NFE各17帧局部续写可行；画质/动作PARTIAL，自由生成长视频与独立泛化NOT_TESTED。166DiT（152生成采样+8回归探针+6commit）、14text、8decode、0训练，2052.980GPU秒。工业场景RGB41–42背景/视角变化明确保留。
- EXP-020：Original原生30NFE全56帧参考工程PASS，天空放射伪影，质量/切换PARTIAL。30DiT，282.237GPU秒，0训练。与GT-history续写条件不同，不作直接E2E/拓扑因果比较。
- 合计196DiT，2335.217单GPU进程wall秒之和=0.6487GPU小时，低于6600秒上限。无需重复GPU补非关键指标。正式V3冻结。

[EXP-019审核](submission/experiments/EXP-019_v3_mixed_action_validation/judge/FINAL_REVIEW.md) · [EXP-020审核](submission/experiments/EXP-020_original_mixed_action_reference/judge/FINAL_REVIEW.md) · [全部本轮对比视频](submission/report/v3/fm2anyflow/data_preparation/mixed_action_validation/README.md)。旧运行状态与授权保留在实验management_snapshot/before_final，仅供历史追溯。

## 唯一当前可执行任务：EXP-021/v1 CPU准备

由既有Exp Worker执行[完整任务书](submission/experiments/EXP-021_v3_fm_baseline_preparation/taskbook_v1.md)。目标：准备32–128真实clip小数据拟合候选、6–8场景episode隔离评测、普通FM因果适配入口与有限GPU profile预算。数据不足如实报告；先做准备再按实测成本放行训练，不直接承诺大训练。

CPU≤4线程、RAM≤16GiB、磁盘新增≤2GiB、首轮≤30分钟；GPU调用/训练更新预算均为0。交付报告后Judge审核，不重复旧实验，不启动AnyFlow/DMD/SGF+/Memory。SolarWM Stage2-fix相机标准推理仍未执行，本轮仅准备其独立任务草案。

Judge负责验收、任务书和Git发布，Worker负责实现/执行/报告。GPU后续日间09:00后最多3卡，夜间最多8张空闲卡，不能抢占他人进程；具体授权以前述任务书为准。
