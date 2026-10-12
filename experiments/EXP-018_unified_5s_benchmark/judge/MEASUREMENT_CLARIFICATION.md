# EXP-018 统一完整计时：Judge实施口径

2026-10-13。用户要求的是七行完整成本，因此本轮主表采用以下具体口径，取代实现初稿中native行仅加载旧条件fixture的做法。

1. 每行独立子进程，重新加载模型/adapter、现场编码同一停车场I0与持续A文本条件、从噪声生成完整124帧、解码并保存124帧MP4。主时间为launcher测得子进程wall（包括Python启动/导入、准备、生成、保存及退出）。严格5秒的120帧展示转码在主计时之外。
2. Native版本复用同源pipeline.units/unit_runner准备条件，旧full37 fixture只作协议核对，不能替代本次真实编码。不得把未含编码的native cold wall与含编码的legacy cold wall放在同一“完整生成耗时”列。
3. 七行统一ABOT_VRAM_RESERVE_GIB=18及同一空闲L40；V1/V2a的dual anchor、V2a在线RGB重编码、V0联合audio/video、V2b历史重算和V3缓存/多次解码属于各自版本实际成本，应保留。
4. 全流程allocated峰含模型/adapter加载、图文编码、采样、clean commit及VAE；分阶段reset必须取所有阶段最大值。reserved峰缺项时标缺测，不能填0或用局部冒充全程。Torch allocated与NVML不同，CPU KV另列。
5. 子项计时必须写实际边界。原legacy的sampling_loop可能含KV提交和RGB anchor；ready-to-video起止若未统一则不作为主比较列。统一主wall可完整对应124帧本次工作。
6. V3/V4最后块无需供后继读取，遵照冻结主线不做额外final clean commit；V1/V2a原协议保留其8次commit并分别计数。
7. n=1，系统磁盘/CPU和编译缓存状态不可能据单次数据完全消除；报告这一限制，不追加重复跑。测量为七个完整方案在明确设置下的单次观察，不作单独KV机制归因。

修改均在EXP-018独立wrapper内完成，不改旧基线源码/结果。当前V0已启动，以上native修正不要求重跑V0。
