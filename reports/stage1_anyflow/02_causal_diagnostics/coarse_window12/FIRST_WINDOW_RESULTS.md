# 同输入首窗口：12latent恢复响应，5latent仍失败

本页只报告已完成的首窗口，不把运行中的后续窗口写成通过。Original权重、native时间、单I0、30步、full37布局与initial noise相同；第一5latent噪声为同一张量的相同slice。

| 方法/历史标签 | 动作 | 前17RGB flow | 整个当前窗口flow | 当前窗口RGB帧数 |
|---|---|---:|---:|---:|
| matched_first5 | A | -1.168911 | -1.168911 | 17 |
| matched_first5 | D | -1.174091 | -1.174091 | 17 |
| coarse_A | A | +0.285428 | +0.863118 | 39 |
| coarse_A | D | -1.198252 | -1.174217 | 39 |
| coarse_D | A | +0.286364 | +0.863568 | 39 |
| coarse_D | D | -1.198491 | -1.174460 | 39 |

完整窗口的首5latent A−D约0.005，首12latent A−D约2.037。为避免只因39帧比17帧长而误判，另列相同前17帧的原Farneback指标；直接截断解码帧iterator，不二次压缩，原17帧指标逐值重放一致。

12latent可见当前窗口包含39帧，前17RGB也可受该窗口后部latent影响；它的控制/显示粒度与5latent不同，不能据此宣称低延迟已经改善。全部17/39静态图中人物和停车场可辨，12latent的A会转向并产生不同运动，5latent的A/D近同。静态图评审不是实时播放评审。后续两个窗口、同状态velocity geometry和GT局部画面仍待验证。

视频：[5latent A/D](matched_first5/AD_local_forks.mp4)；[12latent A](first_window_snapshot/coarse_A/window0_A.mp4)、[12latent D](first_window_snapshot/coarse_A/window0_D.mp4)。原评测完整receipt写在输出目录，提交包中的first_window_snapshot明确仅为首窗口快照。
