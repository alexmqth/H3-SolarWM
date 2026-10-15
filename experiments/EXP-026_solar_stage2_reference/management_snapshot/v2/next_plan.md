# 当前可执行任务：EXP-026/v2

官方单样本输入受访问权限阻塞；v1已完成CPU配置检查与具体阻塞报告，生成能力NOT_TESTED。Judge独立核实61.73GiB完整transformer在FSDP前搬到单L40，不能直接跑SP8。

现授权原Worker立即完成内存安全加载的最小CPU patch与小模型检查：[v2任务书](submission/experiments/EXP-026_solar_stage2_reference/taskbook_v2.md)。0GPU、0下载，CPU≤8/RSS≤32GiB、45分钟、磁盘≤2GiB。官方checkout保持冻结，patch独立交付。无需等待用户再次催促。

数据权限已请求用户通过官方客户端处理，不能把config PASS写成真实reader或视频通过。[v1审核](submission/experiments/EXP-026_solar_stage2_reference/judge/V1_REVIEW.md)。v2结果再由Judge审核具体GPU预算。

EXP-025已accepted执行，视觉/动作PARTIAL，新增收益未证实；停止自动扩训。最终证据与六组视频已push f4238eb，远端核对一致。
