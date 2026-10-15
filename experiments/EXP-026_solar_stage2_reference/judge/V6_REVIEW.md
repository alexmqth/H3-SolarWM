# EXP-026 plan v6 Judge 审核：输入访问检查收口

2026-10-16 HKT。**本轮受控检查执行通过；官方视频验证仍为 BLOCKED_ACCESS。** 不把检查完成写成模型复现完成。

Judge核对ACCESS_PROBES及实际脚本：每提供方一次GET，固定HF revision的test-index返回401，ModelScope H3 encoder_contract返回403，下载payload 0字节。记录与脚本相符，Judge不重复相同请求。匿名失败不等于已获授权账户也不可访问。Worker报告本地官方输入与标准凭据均不存在；CPU reader无真实输入，未声称通过。

0 GPU、0 forward/codec/训练/视频；EXP-026累计保持737.221GPU秒。v5环境与四卡模型加载验收继续有效，环境无需重编译。

## 决策与ROI

本轮到此收口；不追加GPU授权，不轮询下载、不换等价地址，不扩大到SP8全长路线或Stage 2训练。只有官方授权访问已具备，或提供真实匹配的本地index/support/payload后，才重开单样本准备。官方example仅有schema/reader smoke用途的文档证据，本轮未取得其payload，不断言其具体内容已经验证，更不能拿它冒充正式测试样本。

官方Stage 2参考不是V3动作控制研究的前置条件。它提供原生相机任务的参考，不能隔离训练量、参数容量、SGF或DMD的贡献，也不能替代H3-World人物动作基线。后续主线训练另设研究问题和预算，不因该数据权限继续闲耗时间。

[访问证据](../reference_v6/ACCESS_PROBES.json) · [Worker报告](../reference_v6/WORKER_REPORT.md) · [结构化审核](V6_AUDIT.json)。
