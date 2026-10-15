# 当前可执行任务：EXP-026/v1

2026-10-16 HKT。EXP-025/v1已accepted：训练/证据执行通过，视觉与动作PARTIAL，新增收益未证实；不升级V3，不追加该配置预算。

原Exp Worker立即执行官方SolarWM-H3 Stage2-fix最小输入下载与CPU推理准备。固定1条官方相机样本，补齐support/index/payload，实际检查reader/config，提交精确GPU命令与预算。

- [完整任务书](submission/experiments/EXP-026_solar_stage2_reference/taskbook_v1.md)
- [授权](submission/experiments/EXP-026_solar_stage2_reference/AUTHORIZATION.json)
- 本阶段0GPU，CPU≤8并发/RSS≤96GiB，新增下载≤20GiB、磁盘≤30GiB，90分钟窗口；阻塞即消息Judge，不自动全量下载/训练。
- [EXP-025最终Judge审核](submission/experiments/EXP-025_v3_fm_full_coverage/judge/FINAL_REVIEW.md)
- [EXP-025六组三列视频与成本](submission/report/v3/fm2anyflow/fm_full_coverage/README.md)

任务发布后Judge向固定Worker发执行消息，并分别记录送达与实际开工；CPU完成由Judge审核并下发有限GPU预算，无需用户重复催促。

EXP-026交接：已通过消息工具通知固定Worker，他已确认开工（CPU/输入准备）；当前0GPU，尚未声称官方视频推理完成。
