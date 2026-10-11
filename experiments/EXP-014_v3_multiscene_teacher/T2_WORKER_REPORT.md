# EXP-014/v1 T2 Worker 报告：三张训练初图的 FM30 教师目标

2026-10-11 HKT。按照 [T2 Judge marker](judge/T2_APPROVED.json)，三张固定 train 初图在 GPU0/1/2 并行运行，使用同一冻结 V3 FM30 协议：native Single I0、strict chunk-causal、Global RoPE、12-latent C1 + 5-latent C2、sigma0 clean raw KV commit、同一自生成 C1 下 AA/AD 反事实分叉。三图均完成 C1 39 帧及 AA 56 帧；**只有 `s2_9dc2e588` 完成 AD 56 帧**。`s1_7199292c` 与 `s3_b784d995` 的 AD 进程收到 SIGTERM，原工具会话退出码均为 **143**，没有完整 AD endpoint 或视频。本轮新增训练更新为 **0**。

## 实际执行与可用性

| 场景 | 物理 GPU | 已保存的采样 / commit / decode | 运行状态 | 教师目标判断 |
|---|---:|---:|---|---|
| `s1_7199292c` 草地道路 | 0 | 账本预留 78 sampling + 1 commit；已完成 C1+AA 共60 sampling、2 decode | AD 预留18次，结果行只持久化17次后 SIGTERM；无 AD 视频 | C1/AA 可审计；**不是完整配对目标** |
| `s2_9dc2e588` 工业道路 | 1 | 90 sampling + 1 commit、3 decode | 正常 exit 0；516.294 GPU秒，峰值 37.971 GiB | 协议 PASS、画面可辨；D 方向未得到清晰反转，动作目标 **PARTIAL** |
| `s3_b784d995` 暗色草地 | 2 | 账本预留 80 sampling + 1 commit；已完成 C1+AA 共60 sampling、2 decode | AD 预留20次，结果行只持久化19次后 SIGTERM；无 AD 视频 | C1/AA 可审计；**不是完整配对目标** |

三个进程均使用 `CUDA_VISIBLE_DEVICES=<对应物理卡>`、`DIFFSYNTH_MODEL_BASE_PATH=$PWD/H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/models`、`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`，从 GWM 根目录执行：

```bash
.venvs/h3world/bin/python -u submission/experiments/EXP-014_v3_multiscene_teacher/run_teacher.py --stage T2 --scene <scene> --gpu <card>
```

标准输出写入原始 `H3-World/outputs/EXP-014_v3_multiscene_teacher/T2_<scene>.log`；命令、来源、seed13、模型和代码 SHA 由 [任务书](taskbook_v1.md)、[config](config.json)、[source manifest](source_manifest.json)、[code manifest](code_manifest.json)与[复制证据清单](artifacts/T2/copied_artifact_manifest.json)固定。三条实际执行的输入为 EXP-013 中确定的不同训练 episode；A/D 是**合成教师控制反事实**，不是录屏 GT 动作。

两条中断进程从账本 `stage_start` 到最后一次预留分别经过至少 **434.867** 与 **435.929** 秒；`gpu_seconds=0` 只是未执行 `stage_stop` 的不完整记录，**不能解释为零资源消耗**。Judge 的[中断观察快照](judge/T2_INTERRUPTION_AUDIT.json)以首次明确确认进程消失的时钟为上界，分别给出 **515.552** 与 **515.555 GPU秒**。连同 s2 的精确 516.294 秒，本轮 T2 实际用时可界定在 **[1387.090, 1547.401] GPU秒**，低于4050秒阶段预算；它不是精确账本值。中断前 `C1+AA` 的60次模型输出已经完成，最后预留的一次 AD forward 是否完成未知，不把预留次数当成已完成推理。没有自动重试，也没有更换样本或调参。

中断原因尚未确定：两个原工具会话报告 exit 143；Python 日志没有 traceback，账本没有 `stage_stop`。Judge 的 [cgroup 快照](judge/T2_INTERRUPTION_CGROUP_SNAPSHOT.json)显示相关层级 `oom/oom_kill=0`，因此**不能归因为已证实 OOM**。内核日志在当前权限下不可读。原始半途 `AD/result.json`、账本和日志均未覆盖。

## CPU 审计与视觉判断

`s2` 的 [完整 CPU 协议审计](artifacts/teacher_audits/s2_9dc2e588.json) PASS：50层缓存各只有 `index=0`，raw KV 6,799,104,000 bytes；AA/AD 同 C1、同初始噪声、同 sigma、同 Global position，只改第12–16个 action span；旧39 RGB逐值不变。两个 56 帧视频、两个新17帧视频均为832×480/24FPS、可完整解码且 PTS 唯一。AA/AD 新17帧平均像素绝对差为 **10.098**，说明输出不同，但不证明动作语义正确。[同历史并排视频](artifacts/teacher_comparisons/s2_9dc2e588_AA_vs_AD_56.mp4)和[原分辨率关键帧](artifacts/T2/s2_9dc2e588/inspection_keyframes.jpg)可直接查看。

| `s2` C2 | 中央区域 Farneback 水平光流和 | 边界 MAD(38→39) | C2 相邻帧 MAD 均值 | 视觉观察 |
|---|---:|---:|---:|---|
| AA | +68.385 | 10.079 | 12.337 | 人物、道路和工业场景保持可辨；位移明显 |
| AD | +50.074 | 11.173 | 13.132 | 人物仍可辨，姿态/轨迹与 AA 有差异，但场景仍同向运动，未见清晰的 D 反转 |

这是新场景的辅助光流，不是角色动作的真值标签。结合 17 帧联系表及帧39/47/55原分辨率检查，`s2` 支持**局部视觉连续和可检测的分叉**，不支持“该图教师 A/D 方向都正确”的结论。Judge 独立观察也将动作切换判为 PARTIAL；不将其自动纳入合格动作监督集合。

`s1/s3` 使用新增 [中断场景 CPU 审计工具](audit_interrupted_teacher_cpu.py)分别核对了 C1、AA 和实际6.8GB缓存，结果均为 [PASS_C1_AND_AA_ONLY](artifacts/teacher_audits/)：C1 39帧及 AA 56帧可解码、旧39RGB不变、endpoint finite、50层cache各 `index=0`、AA输入prompt/噪声/位置与fixture相符。原分辨率[草地道路 AA 关键帧](artifacts/T2/s1_7199292c/AA/inspection_keyframes.jpg)与[暗色草地 AA 关键帧](artifacts/T2/s3_b784d995/AA/inspection_keyframes.jpg)显示人物及场景在新17帧保持可辨，`s3` 光照较暗。没有 AD 视频，**两图动作差分未评估**，不能因 AA 正常就判其 A/D 目标合格。

## 交付位置与后续处理边界

小型证据、原片副本和逐调用账本在 [`artifacts/T2/`](artifacts/T2/)；大 tensor、原始6.8GB cache及 endpoint 留在 `H3-World/outputs/EXP-014_v3_multiscene_teacher/`，已完成场景的 SHA 由结果行和 `target_manifest.json` 关联。只有 `s2` 生成了完整 target manifest；`s1/s3` 不存在该文件。所有报告和视频是已保存的实际输出，未新训练、未重做 C1/AA。

如果 Judge 决定补齐缺失目标，最小恢复方案是另立 marker 和独立输出目录，重新加载已保存且 SHA 校验通过的 **C1 latent + C1 clean raw KV**，沿原 fixture 的 D action、初始噪声、Global position 和30步 sigma 从 AD 的第一步重新计算，仅执行各图缺失的 **30步 AD + 1次 VAE decode**；因为没有保存中间 noisy latent，不能从第18/20步直接续跑。原 `AD/result.json`、日志和账本必须保留作为失败证据，不能覆盖；新输出再与 C1/AA 做 CPU 等价与视频审计。本报告只提出恢复方法，**没有启动 GPU 恢复**。是否值得补齐由 Judge 根据 `s2` 动作目标 PARTIAL 和训练数据收益决定。
