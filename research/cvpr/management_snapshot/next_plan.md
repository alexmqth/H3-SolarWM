# 当前状态：Action–Video 角色梯度CPU准备已验收

2026-10-17 HKT，Judge。CPU交付已完成并独立复跑通过；本轮授权收口。10DiT/8backward/1800GPU秒只是Worker预算草案，未获执行许可。[验收](submission/research/cvpr/ROLE_GRADIENT_JUDGE_REVIEW.md)。下列任务定义保留为本轮记录，不自动开启新一轮CPU或GPU。

研究问题优先：因果化如何影响已学习的动作—视频交互。前向通路(H1)、历史状态(H2)、优化干扰(H3)分开验证；双Adapter仅候选，不预定最终方法。

[Judge分析](submission/research/cvpr/ACTION_VIDEO_ROLE_REVIEW.md) · [本轮CPU授权](submission/research/cvpr/AUTHORIZATION_ROLE_CPU.json)。前轮证据梳理与提案修订已接受；16DiT/1000GPU秒只是旧提案，未授权执行。

## Task CVPR_ROLE_DIAGNOSTIC_PREP / v1

- Worker：原Exp Worker；基线为冻结Original H3 + released Action LoRA、正确V3及已有EXP-025/027实现。
- 唯一问题：token-role梯度分解能否在现有计算图与LoRA参数空间中被正确测量，以及测量结果能否影响方法选择。
- 交付：`submission/research/cvpr/ROLE_GRADIENT_SPEC.md`、小型CPU梯度分解/LoRA映射验证源码及结果、有限GPU预算草案。只做准备，不加载33B。
- CPU上限：15分钟、4workers、RSS8GiB；0GPU、0下载、0updates、0视频。新授权独立于前轮已收口CPU预算。
- 验证：同一loss下角色贡献加和与autograd相符；包含prefix/other；完整权重与LoRA空间分开；B=0时A梯度退化正确处理；detached历史标明无路径；不把负cos直接判为控制/画质冲突。
- 预算草案：固定少量样本/sigma/层，明确Original-visible与V3可比性、真实动作标签、前后向/commit次数及显存风险。行为对照另明确人物/相机与17帧局限，不启动大矩阵。
- 停止：CPU预算用尽或分解不正确则提交现状；不扩展层数/模型/GPU。Worker不修改冻结生产代码与正式V3结果，不操作Git。

Judge审查后按ROI决定是否值得跑一个有限诊断；相关工作核实与行为证据共同决定novelty，SGF+具体角度/CAER身份尚未独立核实。EXP-027已收口，不追加约1%代理收益训练；仍聚焦ABot，不恢复SolarWM官方支线。
