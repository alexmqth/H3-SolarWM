# EXP-026/v1 Judge 审核

2026-10-16 HKT。接受已完成的CPU准备及阻塞报告；单样本输入准备未完成，GPU/生成能力NOT_TESTED。Worker任务状态blocked，不能把config解析当官方复现通过。

独立核对官方与候选config SHA、官方loader解析、SP8/DP1与sample_count=1、base index 66,280,430,080字节，以及launch --execute退出2且未初始化CUDA。源码optional.py:124–125对整个transformer调用.to(device)，runtime.py:1399加载后才wrap_h3_fsdp；61.728GiB权重超过44.988GiB L40，当前路径不可直接启动。详见V1_AUDIT.json。

HTTP401/403是Worker实际访问报告，Judge未再次请求远端；prepare_cpu.py把当时结果写入manifest是快照，不能称每次重跑都重新验证访问。输入sample_id/reader/hash缺失如实保留。数据权限已向用户说明；不索取聊天明文token。

v1证据冻结在../v1_frozen/。授权v2只做不依赖数据的加载路径CPU修复准备，不授权GPU。GPU成本和视频质量仍未知，原50forward推算不能当实测。
