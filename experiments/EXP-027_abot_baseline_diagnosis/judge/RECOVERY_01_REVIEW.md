# EXP-027 首次训练前失败审核与恢复决定

独立检查GPU_RUN traceback：run_exp027.py audit在coverage.fm_step的GT C1 commit被index0触发；冻结coverage先commit0再forward/backward1，符合协议。此次0backward/update。ZERO_ADAPTER_REGRESSION三路max_abs均0，KV SHA相同；PROBE0已完整。GPU0查询0MiB。

允许一次最小入口修复，恢复复用初始证据，无新增训练方向/数据/超参。原51次调用含失败commit保留，累计上限337；时间总6000秒不增加，恢复上限5500秒。CPU必须测试实际production helper合法/非法路由与checkpoint/RNG/输出路径，后自动恢复；真正训练梯度仍待验证。见补充01。
