# EXP-023/v1 补充03：ffmpeg写视频修复与一次受限恢复

2026-10-15 HKT，Judge批准Worker立即修复并恢复执行。补充01/02继续有效，训练前真实backward门控不变。

## 已确认故障与保留证据

34clip编码完成：70VAE encode、151text、1734.528GPU秒。首次eval_pre的checkpoint0入口回归通过，150项KV/RoPE与velocity exact；首条train_constant GT-history已完成1commit+8采样+1decode，调用系统名ffmpeg时FileNotFoundError停止。首次eval_pre耗时301.064秒、13DiT（4回归+9首片）、1decode、0backward/update。此为媒体保存故障，不能记作模型数值失败或成功视频。

冻结失败日志/账本、当时runner/manifest及checkpoint000，记录SHA至attempt_00归档；原始FAILED_STOPPED状态保持。恢复使用新attempt与新的来源revision，不能抹去失败成本或直接把失败改为完成。

## Worker立即执行

1. 优先检查是否保存了本次首片的完整RGB或latent；若有匹配来源和SHA的RGB则CPU编码恢复，若有latent则仅补必要decode；确认无可恢复输出后，允许**一次**同clip/同checkpoint000/同噪声/同动作的首片重做，最多额外9DiT和1decode。不能为了画质换seed。
2. fm_run及finalize_cpu统一显式使用imageio_ffmpeg.get_ffmpeg_exe()的绝对路径，记录真实路径、版本与SHA；不依赖系统PATH，不安装或修改全局ffmpeg。GPU前在实际写视频函数上用CPU源帧测试单片与并排片的写入、完整解码/帧数/尺寸/fps；codec测试不调用模型。
3. 此后所有生成结果在启动ffmpeg前原子保存已解码RGB（或能恢复的端点及所需history），保存配置/checkpoint/noise身份；写视频临时文件完整解码通过再原子rename。输出完整前保留恢复材料，防止媒体错误再次要求模型重算。仍受磁盘上限约束。
4. 复用已冻结checkpoint000和输入，不重新随机初始化覆盖。回归证据可在模型/attention/输入代码SHA未变且补丁仅媒体/恢复/预算逻辑时复用；若因实现约束需要复跑最多4回归forward，计入原440上限，禁止重做编码。保留新旧manifest与具体diff，严格限定修补范围，不暗改训练数学/梯度门控/采样协议。
5. 预算汇总必须把失败attempt和恢复attempt相加：eval_pre累计时间/调用不得归零；后续训练/post从完整累计账本读限额。phase状态与attempt状态分别保存，失败不是COMPLETE。CPU测试过后Worker直接恢复固定pre-eval→训练→post-eval，按原授权自动衔接，不再等待marker。

## 最小预算变更

**仅VAE decode总上限12→13。** 总DiT仍≤440，128backward/updates、70VAE encode、600text、13800GPU秒、编码4200/训练6000/评测3600GPU秒及GPU/内存/磁盘限制均不变。首次失败13DiT/1decode/301.064秒必须计入。单次首片重复的9DiT使用原有回归余量；不能把预测429当完整账本，因为首次4回归也要计入。完整计数按实际事件汇总。

本补充仅授权一次媒体修复后的恢复。再有模型/协议/梯度/OOM故障或预算耗尽按原硬停；若再次仅媒体保存失败，优先CPU恢复已有RGB，不自动重复模型调用。完成后报告修复、恢复方式、失败费用及最终12条有效评测结果。
