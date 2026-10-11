# EXP-014/v2 Worker GPU 报告：补齐两条中断 AD

2026-10-11 HKT。Judge [恢复 marker](judge/RECOVERY_APPROVED.json)绑定 `recovery_code_manifest.json` SHA `b294c0d1a55a81c374a0c1741f50e59127bad1c4102c0211a55ca0bef7c416f4` 后，按指定顺序在空闲 GPU0 **串行**执行 `s1_7199292c`、`s3_b784d995`。两个原命令会话均 **exit 0**，PID 分别为 `4073273`、`4085917`。每图只从原先保存的 C1 latent、50层sigma0 raw KV和原fixture噪声重新计算缺失 AD 的30步，不重算C1/AA，不做commit、编码或训练。两图原始半途AD目录、账本与原有cache保持不变。

| 恢复图 | GPU0账本 | 墙钟GPU秒 | 峰值allocated | CPU审计 | 动作/视觉观察 |
|---|---|---:|---:|---|---|
| `s1_7199292c` 草地道路 | 30 sampling、1 decode、0 commit/update | 159.631 | 38.178 GiB | PASS | AA/AD分叉明显，AD反向较清楚；AA净位移很弱且途中变号。人物/道路保持可辨，质量与持续A控制为PARTIAL。 |
| `s3_b784d995` 暗色草地 | 30 sampling、1 decode、0 commit/update | 159.061 | 38.178 GiB | PASS | 两路视频过于接近，未见D反向；人物基本存在但场景偏暗。该图**不通过动作目标**。 |

新增总计 **60 sampling forward、2 VAE decode、318.692 GPU秒**，在v2的60F/2decode/1200秒上限内。原v1两个中断的前向与用时仍按[中断审计](judge/T2_INTERRUPTION_AUDIT.json)计费，不因恢复成功而抹掉。连同T1和T2，教师生成总墙钟GPU用时可界定为约 **[2238.043, 2398.354] 秒**（T2中断有上下界）；P1编码另为110.214秒。全任务预记账上限 **402 denoiser forward、12 decode**，其中包含两个中断末尾仅预留、是否完成未知的forward；不能把402写成确认完成的前向数。训练更新仍为0。

Worker 的 [s1 CPU审计](artifacts/recovery_v2_audits/s1_7199292c.json)和[s3 CPU审计](artifacts/recovery_v2_audits/s3_b784d995.json)均验证：原 C1、AA、cache、半途 AD 行 SHA未变；恢复使用原 AD prompt、Global position、初始噪声和31点native sigma；原39 RGB逐值不变；新endpoint有限、shape `[1,24,5,30,52]`；56帧与17帧MP4均为832×480/24FPS、全帧可解码、PTS唯一。Judge 另有[独立s1协议审计](judge/RECOVERY_s1_7199292c_AUDIT.json)与[独立s3协议审计](judge/RECOVERY_s3_b784d995_AUDIT.json)。新目标索引保存在独立 `recovery_v2/<scene>/target_manifest.json`，其 `quality_status=pending_judge`，只是端点来源清单，不代表动作质量通过。

| 场景 C2 | AA 光流和 | AD 光流和 | 新17帧 AA/AD 像素MAD | 解释 |
|---|---:|---:|---:|---|
| `s1` | +3.654 | −45.314 | 24.179 | D分支与A有明显差异，但A持续方向不稳。 |
| `s3` | +66.178 | +58.664 | 3.188 | 两路几乎同向，不能将AD当作正确反向监督。 |

Farneback中央水平光流是场景运动的辅助量，不能单独证明人物语义。以上判断结合全部新增17帧联系表与原分辨率关键帧；[s1关键帧](artifacts/recovery_v2_keyframes/s1_7199292c_AA_AD_keyframes.jpg)和[s3关键帧](artifacts/recovery_v2_keyframes/s3_b784d995_AA_AD_keyframes.jpg)可复核人物结构和分叉程度。[s1 AA/AD并排56帧](artifacts/recovery_v2_comparisons/s1_7199292c_AA_vs_recovered_AD_56.mp4)、[s3 AA/AD并排56帧](artifacts/recovery_v2_comparisons/s3_b784d995_AA_vs_recovered_AD_56.mp4)来自真实保存的H3视频，均经过完整解码检查。原片、日志、逐调用账本副本及来源SHA见[恢复证据目录](artifacts/recovery_v2/)；大latent和cache仍在原 `H3-World/outputs/EXP-014_v3_multiscene_teacher/`。

**研究结论只到教师候选质量分层。** 当前四图中，`s0` 有有限可用的A/D反事实，`s1` 有D反向但A持续性弱，`s2` 和`s3` 都未给出清楚的D反向。它们不能不加筛选地组成“动作正确的四图教师集”；尤其不能用`s2/s3`的AD端点训练并声称已恢复跨场景动作控制。本任务不授权根据这些结果调动作、增scene/seed、训练AnyFlow或做DMD。是否继续多场景学生训练，需Judge按这些真实质量边界另作决定。

逐图的Worker用途标签另存于[教师目标质量清单](WORKER_TARGET_ASSESSMENT.json)，所有成功/中断阶段的预留前向、已确认前向与资源上下界另存于[累计账本](WORKER_CUMULATIVE_BUDGET.json)。两份文件均明确标记为待Judge最终审查，不能替代原始ledger或正式验收。
