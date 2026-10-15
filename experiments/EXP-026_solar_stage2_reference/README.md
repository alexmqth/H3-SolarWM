# EXP-026：官方SolarWM Stage2-fix参考复现准备

最新：plan v4依赖受阻收口，官方复现未完成。固定158帧四卡SP4/DP1配置可行，v3真实NCCL与shard4组已建立，但FlashAttention ABI错误使FSDP/EMA未到。v4官方wheel仍导入失败，现有nvcc13.0与torch cu12.4不匹配，CPU门控阻止第二次GPU尝试。

累计105.265GPU秒，0去噪forward/训练/视频；不能将此解释为四卡显存不足。新任务需解决匹配环境及官方数据权限。

[最新Judge审核](judge/V4_REVIEW.md) · [v4 Worker报告](memory_load_v4/WORKER_REPORT.md) · [v3真实失败日志](memory_load_v3/WORKER_REPORT.md) · [四卡配置更正](judge/FOUR_GPU_CORRECTION.md) · [v2 CPU补丁](memory_load_v2/WORKER_REPORT.md)。

大型官方wheel、overlay及patched源码副本仅本地；依赖receipt/SHA和补丁可复现，不把二进制环境纳入Git。plan版本v1–v4不是研究主线V1–V4。
