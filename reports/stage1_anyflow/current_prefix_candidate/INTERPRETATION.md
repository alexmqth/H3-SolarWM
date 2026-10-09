# 跨chunk结果解释：恢复首块Original不等于恢复历史条件动作场

2026-10-08 22:50。真实33B冻结权重，18状态、120 current/reference前向+16clean-history前向，无训练。完整原始收据和[逐组结果](RESULTS.md)已归档。

| 对照范围 | 原causal delta cosine | own/current-prefix候选 | 结论 |
|---|---:|---:|---|
| 首块6点 | −0.019933 | 1.000000 | 匹配条件下Original身份正控，逐输出相同 |
| 后续12点 | 0.028777 | 0.008548 | 没有恢复动作差分方向 |

后续动作差分范数比从0.6716变为1.0330，relative error从1.1952升至1.4432。幅度接近不等于方向正确。12点中7点cosine提高，但整体仍近正交，不能把局部改善或全18点平均0.3390当作修复成功；后者被首块6个identity大幅抬高。

两个causal分支由同一原始权重分别重建自己的历史KV。每个A/D pair固定raw history、noisy state、anchor、prompt其余部分，均只读cache。相同前向重复RMSE=0；当前A/D改动不污染过去clean-commit KV；模型参数version没有变化。现行causal重跑与此前baseline的每点delta cosine差<1e−8。新的probe保存anchor、teacher output完整hash。

这支持一个受限结论：公共prefix/current-video反馈与own action绑定能解释无历史首块的失配，但组合恢复仍不能让persistent-history的action-conditioned field接近Original。Teacher在当前条件下双向重算整个历史，student固定过去clean-commit表示，这个依赖图差异仍存在。不能因低cosine就断言某个action row错位、漏传或某个权重层唯一有错；也不能断言所有causal方法必然失败。

本候选没有跑生成视频，不能对它的实际光流/视觉质量作结论；没有接入生产默认，没有新增optimizer，没有替换meeting。先不据此开展训练/长视频。准备过的单次rollout入口未启动，保留为源码审阅并明确标注为未执行。

与B的实际视频一起解释：FM48在GT和generated固定state的动作差分均未恢复，停车场30/8也无左右控制，第二自然场景后段人物分解。参见[完整FM48验收](../real_abot_fm/FM48_COMPLETE_REVIEW.md)。

机械正确、训练完成、实际动作/画质通过是三个独立条件。目前前两项有证据，第三项没有通过。C/D仍需可信局部causal能力；这轮结果不支持直接声称只差Stage2。
