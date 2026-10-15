# 当前状态：EXP-026 plan v4已阻塞收口

2026-10-16 HKT。原Worker已停止本轮，无新GPU尝试或后台重试。Judge接受有限依赖诊断；**真实四卡FSDP/EMA加载目标未完成，视频未执行**。

## 最新结论

- 固定158帧官方backend允许4卡SP4/DP1单样本；八卡不是模型硬性最低要求。
- plan v3真实NCCL/shard4就绪，但FlashAttention ABI错误导致在FSDP前退出，26.316秒/105.265GPU秒。
- plan v4官方匹配标签wheel仍无法导入；系统nvcc13.0与torch cu12.4不匹配。CPU门控拦截，0新增GPU秒，没有编译或修改共享环境。
- [最新Judge审核](submission/experiments/EXP-026_solar_stage2_reference/judge/V4_REVIEW.md) · [Worker报告](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v4/WORKER_REPORT.md)。

## 下一步所需条件

1. 真正可用的FlashAttention二进制或与torch cu12.4匹配的独立开发工具链；后续修复可放用户目录，无须预设修改系统CUDA，但本轮没有安装授权和实测结果。
2. 官方相机数据访问或本地副本：HF401/ModelScope403未解决，真实reader仍缺输入。
3. 条件明确后Judge下发有限新任务并消息原Worker立即执行。本轮不追加v5、不为占满GPU开无关实验，不通过换backend改变比较协议。

EXP-025训练已收口，新增收益未证实，不自动扩训。项目累计EXP-026 GPU费用105.265秒；失败费用保留。
