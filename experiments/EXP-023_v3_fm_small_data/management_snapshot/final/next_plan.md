# 当前可执行任务：EXP-024/v1

EXP-023已完成128updates与固定评测，Judge已验收任务完成；无明确视频收益，不升级V3。旧EXP-022梯度协议PASS撤回保持有效。

已授权固定Exp Worker立即执行 **CPU训练覆盖与容量准备**：量化现有adapter变化，实现并CPU检查C1/C2训练入口，核算全层QKV候选容量，交付一个可执行的有限GPU profile方案。最多4CPU线程、30分钟、额外8GiB；本阶段0GPU/0新训练。

[完整任务书](submission/experiments/EXP-024_v3_fm_coverage_preparation/taskbook_v1.md) · [授权](submission/experiments/EXP-024_v3_fm_coverage_preparation/AUTHORIZATION.json)。任务书为具体执行依据，GPU须Judge另行放行。Worker完成后消息回报，Judge审核、维护状态并push。

交接已完成：消息发送至固定Worker会话01a0ece6-63b6-7f42-8c39-cd3ec39a9c80，Worker回复“EXP-024/v1已开工”。
