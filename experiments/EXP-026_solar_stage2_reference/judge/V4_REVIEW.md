# EXP-026 plan v4 Judge审核：依赖门控受阻

2026-10-16 HKT。接受已执行的有限依赖诊断与阻塞交付，Worker status=blocked；完整四卡加载目标未达成。plan v4与研究版本V4/SGF-DMD无关。

## 独立证据

Judge核对官方wheel大小256,007,522字节与SHA256 1c8d2fb083edea3647688fcd4bb21d4641eb01f50359d8fa6200ecd093ba1c54；在独立overlay实际CPU重现flash_attn导入undefined symbol、Diffusers Flash backend requirements拒绝，以及PyTorch编译版本检查CUDA13.0对12.4的major mismatch；全程CUDA未初始化。详见V4_AUDIT.json。

Worker记录的overlay与原失败binary SHA相同、nm符号新旧ABI冲突，与独立导入错误吻合。不能因wheel文件名标记torch2.6/cxx11abiFALSE就宣布可用。共享torch/venv、官方源码和v3证据未改。没有使用原生attention回退来冒充官方协议。

## 成本及结论

v4下载约244.15MiB官方wheel、新本地目录约1.26GB；未开始源码编译，0GPU尝试/秒、0forward/训练/codec/reader/视频。CPU门控正确拦截，避免第二次GPU浪费。v3仍为26.316秒四卡wall、105.265GPU秒；本EXP累计GPU成本105.265秒，不清零失败开销。

四卡SP4/DP1配置可通过官方backend，v3实际NCCL/shard4拓扑已到；真实FSDP、EMA恢复、内存峰及生成仍NOT_TESTED。这不是四卡OOM结果，也不支持回到“必须八卡”的结论。

## ROI决策

本轮收口，不继续试不同wheel/改torch/换attention或无预算安装系统工具链，不下发v5。已有明确最小环境缺口：真正兼容的FlashAttention binary，或与保留torch cu12.4兼容的独立用户目录CUDA开发工具链用于构建；当前系统只有nvcc13.0。后续可另定环境修复任务，但不能声称本轮已完成加载。

官方相机数据HF401/ModelScope403仍是视频复现的独立阻塞。优先获得数据访问或本地输入，同时确认可用的匹配开发环境；条件明确后再安排一次有意义的加载/视频任务。原Worker已通知停止本轮，无GPU常驻或后台重试。
