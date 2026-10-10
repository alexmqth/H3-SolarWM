# EXP-007/v2 AF1 attempt2授权

## Judge修复重跑授权 — 2026-10-11 02:01 HKT

第一次启动在模型加载前因`sha(__file__)`传入str而退出，0forward/0backward/0update，耗时3.538686514秒。Judge已核对traceback与账本，并确认修复为`Path(path).open`，没有改变模型协议。批准一次attempt2；保持原22forward/4backward/1update上限，累计GPU时间仍≤2700秒，第二次最多2696.461313486秒。保留原输出目录、日志和source manifest；attempt2使用独立输出/日志与冻结manifest，CPU preflight通过后直接执行。不需用户再次批准；第二次失败仍停止交Judge。Worker可增加明确的attempt标识/输出配置及累计预算扣除，不改变训练条件。
