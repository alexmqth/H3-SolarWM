# EXP-023/v1 补充04：BF16身份序列化修复后的最后一次首片重跑

2026-10-15 HKT，Judge已完成CPU独立复核，批准Worker立即恢复。补充01/02的输入和梯度门控、补充03的失败保留与RGB持久化规则继续生效。

## 第二次失败与已完成验证

attempt_01在首片9DiT+1decode后，identity对BF16 initial_noise直接.numpy()导致TypeError；身份计算在RGB保存前，输出再次丢失。该attempt耗时81.93666345GPU秒，状态FAILED_STOPPED保留。两次失败合计22DiT/2decode/383.00087027GPU秒；加编码已花2117.52862005GPU秒、70VAE encode、151text、0backward/updates。

Worker MEDIA_SMOKE_V3使用真实train/dev BF16 condition，完成实际GT/self媒体路径与比较片的CPU写入/解码，并注入identity/codec错误验证RGB保留。Judge独立测试同一实际函数与真实BF16 condition，完整解码、两类故障后CPU恢复和2496×540比较片通过，见judge/MEDIA_RECOVERY_CPU_REVIEW.json。测试片均为source RGB，不是模型生成证据。

已验证修复：decode后先原子persist_rgb，再计算identity；tensor SHA用dtype/shape和连续CPU uint8原始字节，避免BF16直接.numpy或静默转浮点改变身份。ffmpeg使用绝对路径、临时输出校验后rename。已保存RGB的任何后续媒体失败只能CPU恢复，不能重新采样。

## 执行授权

允许attempt_02恢复pre-eval，因前两次均无可恢复输出，固定首片再做最多9DiT+1decode一次；这是本轮最后一次因首片媒体问题重复模型生成的授权。复用已冻结checkpoint000/选样/初噪/动作，不重做编码与已通过入口回归。之后完成其余5条pre、128updates、6条post及固定probe；真实首步backward门控仍须在optimizer前通过，正常阶段直接继续，不等待marker。

来源revision绑定本补充、CPU测试报告和实际修复函数SHA，冻结attempt_00/01的完整代码、日志、manifest和原失败账本；累计账本纳入两次失败，不能覆盖状态或归零。源码只允许上述媒体、恢复、累计预算变更，模型数学不变。若重新改动已验证媒体函数，应CPU复核受影响路径，不自动重复模型调用。

## 最小资源调整

总顶层DiT上限**440→449**（新增最多9次首片恢复），VAE decode上限**13→14**。原计划420研究调用+4已做入口回归+18丢失首片调用预计442，最终以事件实际汇总；449是硬上限，不是必须花完。

13800GPU秒（编码4200/训练6000/评测3600）、128backward/updates、70VAE encode、600text、最多2卡/训练1卡、内存和磁盘限制保持。所有失败时间包含在原总时限和评测时限中。没有额度重置，没有额外样本/seed/训练扩量授权。

若再次发生媒体错误，使用持久化RGB完成CPU恢复；若RGB未能保存，不再追加GPU重跑，交缺失证据与原因，由Judge决定缩减该视频交付。模型/协议/梯度错误、OOM或预算耗尽按原硬停。结束后报告真实损失与已恢复产物，不能把失败写成模型质量问题。
