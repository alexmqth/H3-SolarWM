# 2026-10-17 夜间执行报告（11:50 HKT补交）

## 截止与交付状态

原实验于04:10因单任务一小时上限停止。09:00截止守护记录CUTOFF_ENFORCED，原登记的全部controller/timeout/Python进程已在Judge补审计时确认不存在相同身份的活进程；见[STOP_AUDIT.json](STOP_AUDIT.json)。没有以白天恢复授权追溯修改夜间预算。

**未按时完成09:00的Judge汇报与最终推送。** 监控消息在白天再次交互时延迟呈现；原Worker确有发出启动/首场景/终止事件。此前对用户显示的“仍在候卡”只是02:30时状态，不代表后半夜。此报告如实补交，不声称09:00已完成全部交付。

## 实际研究进展

- EXP-028完成并验收：14DiT/8backward、0updates；有限层/状态未建立V3特有的明显角色梯度冲突，不据此开启双Adapter训练。准确合计进程583.800秒；峰allocated23.753/reserved26.770GiB。
- EXP-029在GPU0于03:10:16启动，04:10:14触及预算停止：235DiT=2commit+233sampling、7decode、0训练。实际进程3597.816秒，进程内账本3587.434秒；峰allocated25.441/reserved26.531GiB，RSS83.536GiB。共享显卡与CPU卸载影响吞吐，不能把分支耗时当纯算法对照。
- 已保存s0的ON_A/ON_D/OFF_A/OFF_D四条及对比，s3的ON_A/ON_D/OFF_A三条。s3 OFF_D完成前23次sampling后停止，没有保存端点或视频；其初始RESULT.json中的0不是实际调用数。
- 两项夜间实验合计约4181.616单卡进程秒（1.162 GPU小时），包含EXP-028最初导入失败；不含白天恢复。
- SGF+原文已确认包含双LoRA实现选项；GameWAM动作随机源干预与本项目的外部语言动作路径干预已区分。当前仍无已确证的CVPR新方法。

## 七条输出的初步Judge观察

Judge已查看七张包含边界与全部17新增帧的联系表。山地场景保留可辨人物及腿部运动；ON/OFF均存在A/D分叉，prefix开关改变动作姿态/背景轨迹，不能解释成关闭该边就完全失去控制。暗草地ON_A/ON_D相近、人物较暗；OFF_A有姿态/场景差别，但缺OFF_D，不能完成该场景的2×2判断。尚不作完整行为验收或能力升级。

## 下一步与白天授权

用户11:47明确要求尝试GPU1/6，Judge另行授权[EXP-029有限恢复](../../experiments/EXP-029_current_prefix_behavior/ADDENDUM_03_FINITE_RECOVERY.md)：只补s3 OFF_D，单卡≤1200秒、31DiT/1decode，七条完整结果不重跑。该恢复属于新授权，不属于已经截止的夜间任务。完成后审核全部八条、汇总真实成本并发布视频导航；不启动双Adapter/AnyFlow/DMD。

## 白天恢复最终补记

GPU1恢复于11:51:01–11:56:23 HKT成功完成并退出：31DiT/1decode、321.700秒，峰allocated16.679/reserved19.174GiB。八支结果已齐，Judge已验收有限证据；当前无新GPU授权。该成本不混入上述夜间小计。见[最终审核](../../experiments/EXP-029_current_prefix_behavior/judge/FINAL_REVIEW.md)与[完整视频](../../report/v3/fm2anyflow/current_prefix_behavior/README.md)。
