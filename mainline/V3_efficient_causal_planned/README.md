# V3 · Efficient Causal H3-World — Planned Unification

1. **Version Name / Research Objective**：统一 Efficient Causal Generation + Visual Stability + Action Fidelity。
2. **Parent Version / Baseline**：V1问题、V2a视觉/缓存经验、V2b持续A/D124帧可行性证据；没有直接继承一个已合格的checkpoint。
3. **Main Changes**：研究strict chunk-causal下保留原动作信息流，明确public prefix、历史时间与action routing，配合必要causal adaptation和正确persistent KV。
4. **Model and Inference Configuration**：尚未确定。先30step局部可信，再历史复用与多块，之后AnyFlow、on-policy DMD。
5. **Representative Videos**：**NOT_YET_VALIDATED，无V3视频**。
6. **Quantitative Results**：无V3模型性能结果。
7. **What Was Improved**：已有两支证据明确能力取舍和受控诊断方向，不是V3已改善。
8. **What Still Failed**：strict causal + KV尚未通过动作、视觉与长时效率联合验收。
9. **Lessons Learned**：不能简单拼接两个checkpoint/anchor；核心是让模型在不依赖历史与当前双向重算时仍有正确动作条件能力。
10. **Source / Checkpoint / References**：无V3 checkpoint。[V2a与V2b路线](../../report/roadmap.md) · [验收条件](../../report/02_next_steps/README.md)。

V2a的124f视觉证据和V2b的124f持续动作证据不能直接相加成一个成功模型。本次不启动新实验。

当前按可行性阶段验收，允许普通画质缺陷；有限预算下无有效信号的方向停止。下一候选采用native条件与真实KV短视频，不把V2b改名为V3；任务以根next_plan.md为准。
