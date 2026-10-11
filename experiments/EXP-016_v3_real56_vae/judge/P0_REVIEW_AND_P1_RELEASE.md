# EXP-016 P0审核与P1放行

2026-10-11 08:44 HKT。Judge已读实际入口、预算/时间门与模型加载代码，CPU实际runtime三个模块SHA和冻结源身份独立核对PASS；EXP-015独立数据审计加本轮native preprocess CPU张量确认四图224帧和首39同源、float32[0,1]。P0原权重SHA检查通过，无marker拒绝发生在CUDA/模型导入前，输出目录不存在。

P1_APPROVED.json绑定当前config/source/code摘要，授权仅GPU0实际空闲时单卡一次执行：4image/8video encode/4decode，600GPU秒、44GiB、0text/DiT/训练。08:50前启动，08:57禁止新增调用，失败/显著前缀差异按任务书停止。计算bf16，输出latent dtype按native实测记录，不偷偷cast。

这是模型调用授权，不是编码/重建验收；真实数值、视频和成本待实际结果。不得自动重试或修补action packed协议。
