# EXP-013/v1 Worker 报告：多场景数据审计与下一阶段输入设计

2026-10-11 HKT。`task_id=EXP-013` · `plan_version=1` · `worker_status=complete_pending_judge`。按 [任务书](taskbook_v1.md)只执行 CPU 读写和原有视频解码；**GPU 调用 0、VAE 编码 0、denoiser forward 0、训练更新 0、下载 0**。本任务不声称多场景 AnyFlow 训练或视频质量改善。

## 可复跑方法与原始证据

在 GWM 根目录运行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python -u \
  submission/experiments/EXP-013_v3_multiscene_data_plan/audit_data.py \
  > submission/experiments/EXP-013_v3_multiscene_data_plan/audit_run.log 2>&1
```

脚本读取本地 `H3-World/data/abot_bridge/{clips.jsonl,annotation_pilot.json,preparation.json,encoding.json,encoded_manifest.json,source_metadata/metadata.jsonl}`、六组 `raw/data/*/*/{video.mp4,annotations.tar}` 与24组三件套 `{clip.mp4,action.npy,first_frame.png}`。它完整 CPU 解码24条 clip，逐项重新哈希，调用原 `code/abot/abot_action.py` 从 annotations 重建每帧 17 列动作，并用 PyAV 检查帧数、FPS、PTS、尺寸和 PNG 是否逐像素等于解码首帧。输出 [逐 clip inventory](inventory.json)、[源 manifest](source_manifest.json)、[source/split/action 审计](source_split_action_audit.json)、[候选初图 manifest](candidate_manifest.json)、[CPU 联系表](candidate_contact_sheet.jpg)、[运行日志](audit_run.log)。本地另保留同内容 JSONL，仓库 `.gitignore` 会忽略其扩展名。源大文件仍在原路径，未复制或覆盖。

## 审计结果

- 实际 **24 clips / 6 episodes**：train 16 clips/4 episodes，validation 8 clips/2 episodes。两个 validation episode 恰好是 EXP-011/012 已观察的工业 `118eb5d8…` 和村落 `dfec8ed3…`；它们完整排除未来训练，只能叫固定回归评估，不能叫新的盲测。train/validation 的 episode ID、源视频 SHA-256、annotations SHA-256 均无交叉。同 episode 多 clip 不独立；本组24片源帧区间在同 episode 内也没有实际重叠。
- 24/24 clip 完整解码为 **39帧、24FPS、832×480、单调等间隔 PTS**；24/24 PNG 与已解码第0帧逐像素一致。原视频各 30FPS、1920×1080、1800帧；六个视频及 annotations 哈希与 `clips.jsonl` 一致。所有 clip/action/PNG 自身 SHA 也与 manifest 一致。24/24 `source_frame_indices` 符合从 `src_start` 起每5源帧保留前4帧的 30→24FPS 公式。
- 24/24 `action.npy` 是 finite float32 `[39,17]`，前11列为二值按键，后6列为姿态旋转与位移。用真实 annotations/COLMAP pose 和相同 episode 归一化尺度重建，24/24 数组最大绝对差低于 `1e-5`；`caption.scene_static` 与 manifest 的 prompt 逐项相同。动作列语义见本地原生成代码 `H3-World/code/abot/abot_action.py`：`W,A,S,D,Q,E,I,J,K,L,Space,d_pitch,d_yaw,d_roll,d_x_right,d_y_down,d_z_fwd`，而不是文件名的单一字母。
- 例：`b784…_A_1750` 的39帧全部是 **W+A**；`b784…_D_1205` 全部是 **S+D**。四张训练候选中，真实录屏分别为 **A+S+L、A+S、A+S+J、纯A**。文件名 A 只是目标键在片段中持续存在，不能当“纯 A”动作 GT；之后 teacher 的 A/D 是明确构造的模型反事实控制，不能假装复现这些真实录屏。
- 24个原源首帧做了 `n−1/n/n+1` 像素探针：21条标称 `n` 的 MAD 最小，3条同一 episode 的邻帧更小或极接近。这个探针受 FFmpeg 缩放/双重压缩影响，**不能据此宣称错位**。对三条逐个按原生产脚本的 `select,setpts,scale,crop,libx264 CRF14` 配方从源视频重建，生成的 MP4 与已存文件 **三条均 SHA-256 完全一致**；源视频到裁剪 MP4 的实际生产路径正确。其余21条通过强度较低的邻帧像素探针；未逐字节重建全部24条。动作与记录索引的精确重建均已通过。
- 旧 `encoded_manifest.json` 的 `anchor_protocol=global_retimed_rgb_prefix_last_image_dual_v2`，不符合 V3 的 native Single I0。现有 39RGB/12latent clip 只覆盖 C1，不包含56帧 C2 GT；若未来使用真实 C2，必须从源片按56帧索引重新剪裁并用对应 source action 重新生成动作，不能把已有39帧 `.pt` 当作 C2。四张选定初图的 `src_start≤1731`，理论上可在1800帧源片内扩到56帧，但本任务没有实际执行。

## 确定性训练初图

每个 train episode 在 `target=A` 片中取最小 `src_start`（同值按 `clip_id`），不按观感选图、不使用 validation。四张候选均满足 A 条件，无 fallback。与 Judge 的独立选图记录逐一一致；审计脚本并未读取 Judge 的期望文件。

| episode 前缀 | clip / 首源帧 | 真实录屏按键（39/39） | 新 teacher 动作 |
|---|---|---|---|
| `43866101` | `A_1245` | A+S+L | 合成 A/D 反事实 |
| `7199292c` | `A_1670` | A+S | 合成 A/D 反事实 |
| `9dc2e588` | `A_1410` | A+S+J | 合成 A/D 反事实 |
| `b784d995` | `A_1635` | A | 合成 A/D 反事实 |

候选 PNG SHA、完整静态 prompt、源标识和真实组合已冻结在 [manifest](candidate_manifest.json)；[联系表](candidate_contact_sheet.jpg)直接标注真实按键。后续即使某图生成难看，也不换图/seed 来挑结果。

## 对下一阶段的判断与边界

这个小数据集足以支撑一次**严格来源可追溯的多初图可行性试验**，不足以证明统计泛化。EXP-007 的停车场训练目标来自另一个冻结 V3 generated 输入；现有来源清单未提供可与六个 ABot episode 严格关联的原始 episode ID/SHA，不能编造“停车场与ABot完全独立”的证明。它仅作已观察回归参考。

[拟议协议](PROPOSED_PROTOCOL.json)和[阶段性资源/停止门](FUTURE_GPU_DESIGN.md)分别明确新 native Single I0 编码、冻结 FM30 teacher、自 V3 初始化的独立多场景 AnyFlow student、固定 FM8/AF8 回归评估。K=4 时教师上限 **364 denoiser forward/12 decode**；建议 student 32 update、**640 forward/128 backward**，每次按当前权重重建四场景 C1 KV，保留单一预定终点。数字是下一任务的建议预算，**不是本轮实测或获批 GPU 工作**。先看教师是否能在四图 A/D 下保留人物与方向；若教师目标本身失败，停止训练准备并交 Judge 判断。AF2 step32 在新场景迁移的负结果不因数据准备完成而改变，DMD 不自动启动。

本轮新增小文件约 0.3 MB，原数据与 checkpoint 均未修改。可复跑脚本成功退出，24 clip 检查无失败；最后仍需 Judge 独立验收和决定是否发布下一 GPU 任务。
