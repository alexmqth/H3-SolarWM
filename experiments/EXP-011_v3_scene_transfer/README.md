# EXP-011/v1 — 两个固定 ABot 初图的 V3 短程迁移

当前状态：**P0/P1、G1 四首窗和 G2 四组 AA/AD 续写均已完成，Judge 最终验收为两块56帧的有限可行、质量 PARTIAL。** 这是两张固定 validation 初图的短程迁移，不代表124帧或跨场景统计泛化。完整视频、动作分叉、缓存审计与预算见 [G2 Worker 报告](G2_WORKER_REPORT.md) 和 [Judge 最终验收](judge/FINAL_REVIEW.md)；[G1 报告](G1_WORKER_REPORT.md)、[P1 报告](P1_WORKER_REPORT.md)、[P0 报告](P0_REPORT.md)保留前置过程。正式目标和预算以 [taskbook_v1.md](taskbook_v1.md) 和 [next_plan.md](../../../next_plan.md) 为准。

输入仅为 source manifest 中两张 832×480 validation PNG 和各 episode 的 `caption.scene_static`。旧 ABot `encoded/*.pt` 使用 RGB dual-anchor 协议，本轮代码不读取。每场景按原生 H3 full37 packed horizon 生成同一 Single I0、seed13 video/audio noise 和 A/D 文本；G1 分别运行 V3 FM30 与普通 FM8 的 39 帧首窗，G2 从各自首窗的 clean KV 分叉 AA/AD 至 56 帧。FM30 仍是 causal V3，不是 Original 双向参考。

代码入口：

```bash
cd /home/qma/work/GWM/submission/experiments/EXP-011_v3_scene_transfer
PY=/home/qma/work/GWM/H3-World/outputs/2026-10-09-16/submission_acceptance/venv/bin/python
$PY encode_native.py --preflight
$PY test_p0.py
```

GPU 阶段必须先收到 Judge 在 `judge/` 创建的 marker。每个 marker JSON 至少包含 `task: EXP-011/v1`、`stage`、`approved: true`、`config_sha256`、`source_manifest_sha256`、`code_manifest_sha256`，哈希为相应文件本身的 SHA-256。合法 stage 为 `P1`、`G1`、`G2_industrial_FM30` 等四个场景/方法组合。**本目录不自建批准 marker。** 每次只用一张当前空闲卡，物理编号必须和 `CUDA_VISIBLE_DEVICES` 一致，例如：

```bash
CUDA_VISIBLE_DEVICES=0 $PY encode_native.py --encode --gpu 0
$PY audit_fixtures.py
CUDA_VISIBLE_DEVICES=0 $PY run_exp011.py --stage G1 --scene industrial --method FM30 --gpu 0
CUDA_VISIBLE_DEVICES=0 $PY run_exp011.py --stage G2 --scene industrial --method FM30 --gpu 0
```

上面 GPU 命令仅为格式示例；实际放行以 `judge/` 中对应 marker 为准。P1 的实际调用为 text encoder 6、image VAE encode 2、video VAE encode 0、denoiser 0、decode 0。G1 实际76 sampling / 4 decode，G2 实际152 sampling + 4 clean commit / 8 decode；推理总账本232 forward / 12 decode / 1781.710 GPU秒，复制在 [artifacts/G2/budget_final.json](artifacts/G2/budget_final.json)。每次调用前记账，失败也保留；没有自动重试、额外场景或训练。

Judge 已目视四条首窗全部156帧与四组G2的全部新增帧，并在 [最终验收](judge/FINAL_REVIEW.md)中接受短程有限可行/quality PARTIAL。脚本状态 `complete_pending_visual_review` 只表示计算完成，研究结论依 Judge 审核确定。两组视频入口：[工业 FM30 AA/AD](artifacts/G2/comparisons/industrial_FM30_AA_vs_AD_56.mp4)、[工业 FM8 AA/AD](artifacts/G2/comparisons/industrial_FM8_AA_vs_AD_56.mp4)、[村落 FM30 AA/AD](artifacts/G2/comparisons/village_FM30_AA_vs_AD_56.mp4)、[村落 FM8 AA/AD](artifacts/G2/comparisons/village_FM8_AA_vs_AD_56.mp4)。
