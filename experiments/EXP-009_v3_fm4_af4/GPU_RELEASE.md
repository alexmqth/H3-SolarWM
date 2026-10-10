# EXP-009/v1 Judge C2 GPU放行

2026-10-11约04:41 HKT。独立重跑CPU preflight通过，28项来源SHA及AF2 step32配对权重一致。原生4-step sigma为1、0.8694517212、0.6894410400、0.4252873230、0；FM4使用冻结interval_cached和原生scheduler，AF4使用原student入口与next-sigma finite map。更新公式FP32顺序差异上限1.1921e-7，已记录，不改协议。

现在批准GPU0顺序执行两个模型各prefill→AA C2→AD C2：本阶段最多22forward/4VAE/0update，累计任务预算仍38forward/8VAE/.60GPUh，allocated≤44GiB，截止09:00。启动前核对空闲；全任务共享账本和互斥锁。每块失败不自动重试，持续噪声/主体场景崩溃停止该候选后续启动；普通画质缺陷按PARTIAL。第三块尚未批准，需Judge看C2全部新增帧后更新主marker并发独立C3 marker。

源码与输入绑定GPU_AUTHORIZATION.json的manifest SHA；不得在运行期间修改冻结runner/config/依赖。后续只准备CPU视频对比与报告可继续。正式基线不覆盖。
