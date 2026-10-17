# EXP-029 · Current-prefix feedback × current action

**状态：**[Judge 已验收有限的行为对照证据](../../../../experiments/EXP-029_current_prefix_behavior/judge/FINAL_REVIEW.md)；没有产生新模型版本，也没有后续 GPU 任务授权。

**要看什么：**同一段由 V3 自己生成的 39 帧历史、同一首帧/噪声/缓存，仅改变当前动作 A（左移）/D（右移），以及当前非动作 prefix 是否直接读取当前视频。这里 prefix 指 packed sequence 中的视频前条件 token，包含静态语义、初始图像及单独的 action spans；本实验开关只作用于非动作查询。ON 保留已验收的 V3 current-prefix feedback；OFF 只删除这一条 attention 边。两种配置都保留 current action 与 video 的原有路由。模型仍是 Original H3 + released Action LoRA，`h3_fp32`、CPU raw KV、30 NFE/块，没有新增训练或 AnyFlow。

**缩写：**NFE（Network Function Evaluations）是网络求值次数，30 NFE 表示每个新块进行30次采样前向，不含另记的历史 commit（将干净历史写入缓存）。KV 指 attention 的 Key/Value 缓存；DiT 指视频去噪 Transformer。A/D 的完整条件分别是人物向左/右横移，并要求相机跟随。

先看两条 2×2 视频。每片 **56 帧 = 39 帧保存的自生成历史 + 17 帧本实验新生成续写**，24 fps，832×480。比较视频按行 ON/OFF、按列 A/D 排列，前39帧在四格内完全相同。

| 场景 | 2×2 对比 | 观察重点 |
| --- | --- | --- |
| s0 山路 | [ON/OFF × A/D](videos/s0_ON_OFF_AD_2x2_56.mp4) | ON 下 A/D 轨迹明显分叉；OFF 仍有分叉，但 A 的水平画面运动减弱。人物和山路在新增17帧内连续。 |
| s3 草地 | [ON/OFF × A/D](videos/s3_ON_OFF_AD_2x2_56.mp4) | 四条续写较相似；不能据此声称 A/D 控制已恢复。人物较小，画面整体运动受相机影响。 |

单条 56 帧原片：

| Scene | ON_A | ON_D | OFF_A | OFF_D |
| --- | --- | --- | --- | --- |
| s0 | [视频](videos/s0_ON_A_rollout56.mp4) | [视频](videos/s0_ON_D_rollout56.mp4) | [视频](videos/s0_OFF_A_rollout56.mp4) | [视频](videos/s0_OFF_D_rollout56.mp4) |
| s3 | [视频](videos/s3_ON_A_rollout56.mp4) | [视频](videos/s3_ON_D_rollout56.mp4) | [视频](videos/s3_OFF_A_rollout56.mp4) | [视频](videos/s3_OFF_D_rollout56.mp4) |

全部新增帧加块边界的联系表位于 [contacts/](contacts/)；[video_manifest.json](video_manifest.json) 给出每个汇报副本的原始文件和 SHA。s0/s3 的原始 ON 分支与此前 EXP-014 完整视频的 raw56 RGB **逐值相同**，见 [CPU replay audit](ON_HISTORICAL_REPLAY_CPU.json)。s3 OFF_D 因原夜间进程达到1小时上限未保存视频，已用同一 C1 和完全一致的 raw KV，在单次 [finite recovery](../../../../experiments/EXP-029_current_prefix_behavior/recovery01/ALL_EIGHT_MANIFEST.json) 中补齐；其他七条没有重跑。

| 场景 | A/D 新增17帧 RGB MAD：ON → OFF | 中央 Farneback 水平光流：ON_A / ON_D | 中央 Farneback 水平光流：OFF_A / OFF_D |
| --- | ---: | ---: | ---: |
| s0 | 28.31 → 20.93 | +42.15 / −27.46 | +16.25 / −26.86 |
| s3 | 3.19 → 3.16 | +66.18 / +58.66 | +77.95 / +78.11 |

| 执行 | 完整进程时间 | DiT 调用 | 解码 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 夜间原运行，GPU0 | 3597.816 秒 | 235（2次 commit + 233次采样） | 7 | 七支完成，s3 OFF_D 在第23次采样后到时停止 |
| 单支恢复，GPU1 | 321.700 秒 | 31（1次 commit + 30次采样） | 1 | 仅补 s3 OFF_D，成功退出 |
| 合计实际消耗 | **3919.516 单卡秒** | **266** | **8** | 八支完整续写；没有训练更新 |

缓存占用为每个 C1 **6.332 GiB CPU raw KV**；原运行 GPU allocated 峰25.441 GiB，恢复运行峰16.679 GiB。两次运行共享卡负载不同，不能把耗时或显存差异解释成 ON/OFF 加速。原运行的23次未完成分支采样仍计入合计成本。

MAD 衡量分支画面差异，**不是动作正确率**。Farneback 是中心区域屏幕运动代理，包含相机运动；s3 的同号结果不能证明角色向 A/D 两侧移动。完整边界 MAD、帧内 MAD、采样/解码时间、逐帧 flow 与资源账本见 [Worker report](../../../../experiments/EXP-029_current_prefix_behavior/report.md)和 [CPU diagnostics](ACTION_DIAGNOSTICS_CPU.json)。当前证据表明这条 prefix 边在 s0 对行为有**可见影响**，但不是跨场景动作保真的充分条件；没有测试124帧长期效果。
