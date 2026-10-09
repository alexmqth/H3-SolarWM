# EXP-002初版实现审阅（在途，非能力验收）

2026-10-10 04:02 HKT，Judge。已读取初版interval_cached.py、run_rollout.py、config.json、test_contract.py及冻结H3缓存实现。

入口显式区分全局start与cache index，支持[0,12)、[12,17)、[17,22)，Single I0与fixed_prefix_timesteps=False；AA/AD从同一first12_A cache文件开始，未来action/video经visible_inputs裁剪。prefill一次后缓存保存并复用，可减少重复首窗开销。上述是源码检查，不代表实际视频或cache数值验收。

开跑前已通知Worker两项局部修正：sampling前后检查实际内存cache的entry/tensor身份、version与计数，不能用磁盘文件hash代替；记录并限制含prefill/commit/VAE的整体GPU峰值，不能在prefill后reset而漏计。均不要求新GPU实验或扩预算。可复用旧current-prefix CPU祖先重放与未来隔离测试，不重新开展18状态诊断。

等待实现/CPU检查和实际有限视频；普通画质缺陷按可行性范围记录，不额外加成熟画质门槛。


04:03补注：首个AA进程已先行启动，Worker保留run_rollout_aa_at_run.py；不要为补测量重跑GPU。首个AA的磁盘hash检查只按磁盘证据解释，prefill峰值缺项如实披露；后续AD/第三块补内存只读检查与整体峰值。已有9项输入/布局CPU及10项旧current-prefix检查通过，首窗模型身份回放通过，最终视频与能力仍pending。


## AA第二块初审（04:06 HKT）

首条AA已完成56帧，独立完整解码确认56帧/24fps/单调PTS，核对首39已发RGB及完整56帧hash。Judge静态查看RGB38–55全部序列和末帧原尺寸，单体人物/肢体与停车场结构可辨；有姿态节奏变化与细节软化，未见严重分解或换场。新段flow +1.3473，sampling142.91秒；first12模型身份回放relative RMS=0。真实初始KV为50层、各4680历史token，共6,799,104,000 bytes；第二块commit后各新增1950token，共9,632,064,000 bytes。单条A方向并不证明当前动作条件控制已成立，待同history的AD视频与后续自身历史结果。


## 同history AA/AD第二块对照（04:10 HKT）

AD完整56帧已独立解码并验证旧RGB前缀和完整RGB hash；静态查看新增17帧及原尺寸末帧。AD保持单体人物/停车场，可辨运动方向与AA不同；普通细节/姿态节奏限制存在，未见严重崩坏。固定历史、首窗cache、初始noise、anchor、audio、位置与首39RGB摘要均匹配，当前prompt不同，详见second_pair_checks.json。AA/AD flow +1.3473/−1.4577；这对视频支持当前动作条件响应的局部可行性。AD实际内存cache identity/version检查通过，sampling135.69秒。允许按原任务继续两条自己的第三块到73帧，不追加诊断；能力最终验收仍pending。


## AA第三块73帧初审（04:14 HKT）

已完整解码73帧并核对56帧前缀、73帧RGB、endpoint和恢复cache的hash链；sampling内存cache未变。Judge静态查看RGB55–72完整序列与末帧原尺寸，人物保持单体、场景可辨。55→56姿态/位置有明显跳变（boundary灰度MAD15.01），有节奏和细节不足，记录为可行性质量限制；尚未见人物分解或换场。新段flow+0.4738、sampling155.14秒，见AA_third_checks.json。等待AD第三块与最终报告，不要求额外诊断。


## AD第三块与最终交付审阅（04:22 HKT）

AD第三块73帧完整解码、前56帧冻结、完整RGB/endpoint/恢复cache摘要链均通过；实际内存cache identity/version未变。Judge已静态查看RGB55–72全部帧及原尺寸末帧，单体人物与停车场可辨，普通细节软化，未见严重分解/消失；flow −0.9298，边界MAD6.32。见AD_third_checks.json。所有四次进程已关闭，nvidia-smi无计算进程，账本124 forwards/4 VAE/0.22923 GPU-hours。交付manifest全部20项hash匹配，另完整解码两条73帧并排片并查看RGB56布局。未声称正常速度实际播放；正式结论见FINAL_REVIEW.md。
