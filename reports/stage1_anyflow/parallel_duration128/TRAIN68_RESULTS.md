# 真实33B四卡64→68：完成，后续GPU空闲检查拒绝

记录：2026-10-08T13:26:54.078754+08:00。这是训练执行/恢复验证，不是新模型视频验收。

4次新增更新及加载/前后validation共327.30秒；逐更新wall合计176.00秒。GPU3–6各一完整H3副本，全局logical batch仍4，原AnyFlow四forward/样本不变。

| update | action | chunk | wall s | total scaled loss | grad norm |
|---:|---|---:|---:|---:|---:|
| 65 | A | 2 | 62.97 | 0.06078932 | 0.01530837 |
| 66 | D | 2 | 64.11 | 0.09262209 | 0.01320853 |
| 67 | A | 0 | 25.13 | 0.06451534 | 0.03822285 |
| 68 | D | 0 | 23.80 | 0.09402236 | 0.00846691 |

真实预更新checkpoint恢复：四套adapter全部tensor/布局、Adam、step/update history、teacher身份、logical/CPU/CUDA RNG、固定config逐项完全一致。四卡预验证的8个raw loss与原单卡64次后验证完全一致，sigma/r/类型/权重相同。实际训练噪声未单独记录hash，不扩大为每个CUDA中间值逐bit等价的声明。

最终四replica的bank、Adam moments、三类RNG哈希全部一致；四组bank参数都有更新，原visual与target-time冻结。Allocated峰值MiB：GPU3=40504.28, GPU4=40504.67, GPU5=40503.25, GPU6=40503.44。训练reset峰值后测量；总主机内存、四GPU总占用不能与单卡混淆。

这四次更新只包含chunk2/2/0/0，不是覆盖所有chunk的正式吞吐benchmark。先前只读GPU探针与CPU多步/恢复已有独立记录；本轮补上了真实GPU optimizer与replica一致性。CUDA归约次序和单卡不同，不声称与串行后续optimization逐bit相同。

旧队列在68成功之后、96启动之前因GPU3被新任务占用而安全停止。已准备独立resume68_to128控制器接续到96并生成A/D8。数字gate通过则补A/D4并停待完整视觉/匹配FM/独立seed/schedule；否则再到最多128并生成4/8。Stage1尚未通过，没有开始Stage2。

- [真实训练记录](training68.json)
- [真实回载核对](resume64_audit.json)
- [固定验证前后对照](validation_resume64.json)
- [完整固定协议](PLAN.md)
