# EXP-001 · V2b 自身历史持续动作 124 帧

**Worker执行完成，Judge accepted：持续A/D124帧可行性通过，限制保留。** 在单个停车场场景、seed 13 下，持续 A 和持续 D 两条自身历史路径均生成 124 RGB 帧，人物和场景保持可辨，A/D 水平光流方向持续不同。此前 v2 四路径门槛在 DA 第三块失败的事实保留：AD/DA 只到 73 帧，本轮不宣称动作切换通过。完整结果请从 [Worker 报告](worker_report.md)、[机器指标](metrics.json)和[视频](artifacts/videos/)进入。

## 真实视频

| 动作 | Original 124 帧 vs V2b 124 帧 | V2b 原片 | 限制 |
| --- | --- | --- | --- |
| A | [并排对照](artifacts/videos/Original_vs_V2b_A_124.mp4) | [AA 124 帧](artifacts/videos/AA_rollout_124.mp4) | RGB72→73 有可见转身/位置跳变 |
| D | [并排对照](artifacts/videos/Original_vs_V2b_D_124.mp4) | [DD 124 帧](artifacts/videos/DD_rollout_124.mp4) | 末段人物接近左下画面边缘 |

[A 后接 D 的 248 帧对照总览](artifacts/videos/Original_vs_V2b_AD_overview_248.mp4)便于开会连续播放。旧四路径证据也完整保留：[AD 73 帧](artifacts/videos/AD_rollout_73.mp4)、[DA 73 帧](artifacts/videos/DA_rollout_73.mp4)和[第三块复核记录](artifacts/S1_interim_review.md)。所有视频来自真实保存的 MP4；并排片左 Original、右 V2b，24 fps。Original 与 V2b 共享首图、seed、视频/音频初始噪声等已核对输入，但完整去噪、历史协议与音频处理不同，是跨协议能力比较，不是单因素消融。

## 采用的推理协议

V2b 没有新增训练 checkpoint：Original H3 + released action LoRA，Single I0、native action/time、全局 RoPE、12-latent 首块接每次 5-latent，30 steps/chunk，flow shift 2.22。当前 chunk 只看到过去和当前动作/视频；每个 sigma 把自身已生成历史临时加噪到同水平，与当前块在可见窗口中联合双向重算，**不使用 persistent hidden KV**，也不重置 GT/teacher 历史。已发布 RGB 只追加不回改。它是局部自回归候选，不等同于已完成 strict chunk-causal 高效模型、AnyFlow 或 DMD。

| 新增 RGB 区间 | AA / A 水平 flow 均值 | DD / D 水平 flow 均值 | AA / DD 边界灰度 MAD |
| --- | ---: | ---: | ---: |
| 56–72 | +0.347 | −0.879 | 4.51 / 9.41 |
| 73–89 | +1.477 | −0.870 | 16.25 / 7.82 |
| 90–106 | +0.685 | −0.611 | 3.22 / 2.30 |
| 107–123 | +0.987 | −1.260 | 3.07 / 2.41 |

全片 124 帧 flow：Original A +1.077、D −1.602；V2b A +1.035、D −1.050。Farneback 仅是固定中心裁剪运动代理，不能替代动作语义和画质判断。静态逐帧接触图覆盖全部新增帧，关键边界已看原尺寸；Judge 另行独立审阅完整视频。AA 的一次明显边界跳变和 DD 的画面边缘漂移需要在演示时直接指出，不能称为像素级稳定或与 Original 等质。

## 成本、完整性与版本记录

本任务全部 4 条第三块及 AA/DD 后续 3 块共 **300 次 sampling + 12 次诊断 = 312 次 denoiser forward**、18 次 VAE decode、0 次训练更新。累计 3704.85 GPU-seconds，即约 1.0291 GPU-hours；单卡峰值 allocated 26392.86 MiB。V2b 完整 124 帧需要 6 块 × 30 = 180 次 sampling forward，且每步重算历史；本次只测到旧 73 帧之后三个新块的增量 wall time：A 1216.8 s、D 1194.9 s。Original 全片 E2E 分别 478.4 s 和 454.6 s，**计时范围不同，不能形成公平加速比**。逐块耗时、显存和原始数值见[metrics.json](metrics.json)。

[CPU 完整性审计](artifacts/video_integrity_audit_v3.json)逐个解码 124/73 帧原片、所有新增 17 帧片及边界片，检查 H.264、24 fps、PTS、帧数、endpoint SHA 与每次已发布 RGB 前缀 hash；全部通过。[三条并排片元数据](artifacts/comparison_manifest.json)也记录源视频和输出 SHA。v1 运行源码/config 与四路 73 帧结果保持原样；v3 独立使用 [runner](run_rollout_v3.py)、[配置](config_v3.yaml)、[冻结来源清单](source_manifest_v3.json)和[90 帧范围门槛](artifacts/raw_json/gate90.json)。旧 v2 指标另存于 [metrics_v2_interim.json](metrics_v2_interim.json)，没有将原四路径 FAIL 改写为 PASS。

可用项目已有环境做 CPU 复核：

```bash
.venvs/h3world/bin/python -m pytest -q submission/experiments/EXP-001_v2b_124/test_v3_contract.py
.venvs/h3world/bin/python submission/experiments/EXP-001_v2b_124/audit_v3_outputs.py
```

大型 latent endpoint、未压缩发布帧和权重保持原位置，详细路径与哈希见 [MANIFEST.md](MANIFEST.md)。本轮并未再启新的训练、AnyFlow、DMD 或 Original GPU 推理。

[Judge正式验收](judge/FINAL_REVIEW.md) · [冻结结案任务书](accepted_taskbook.md)。原开跑前测试/manifest保存在*_at_run文件；当前test_v3_contract.py检查完成态，Judge复核3 passed，生成runner/config保持冻结hash。
