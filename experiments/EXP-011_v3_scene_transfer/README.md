# EXP-011/v1 — 两个固定 ABot 初图的 V3 短程迁移

当前状态：**P0 CPU 准备与 P1 两场景原生输入编码已完成，P1 后 CPU fixture 审计 PASS；G1/G2 尚未放行或执行。** 正式目标和预算以 [taskbook_v1.md](taskbook_v1.md) 为准；P0 事实见 [P0_REPORT.md](P0_REPORT.md)，实际 P1 调用、失败尝试和新 fixture SHA 见 [P1_WORKER_REPORT.md](P1_WORKER_REPORT.md)。

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

上面 GPU 命令是分阶段运行格式，**不能作为当前 G1/G2 放行凭据**；P1 已按第三次 Judge marker 完成。P1 的实际调用为 text encoder 6、image VAE encode 2、video VAE encode 0、denoiser 0、decode 0。G1 四个首窗计划合计 76 sampling forwards / 4 decode；G2 四组 AA/AD 计划合计 152 sampling + 4 clean commits / 8 decode。共同账本保存在 `H3-World/outputs/EXP-011_v3_scene_transfer/budget.json`，每次调用前记账，失败也保留。09:00 HKT 后的 GPU 阶段拒绝启动，空闲磁盘低于 60 GiB 或物理卡空闲显存低于 44,000 MiB 也拒绝。无自动重试，输出目录存在则拒绝覆盖。

G1 完成后须先目视全部 39 帧，再由 Judge 决定每个配置是否进入 G2；G2 的 marker 按场景/方法分别发放。生成质量、动作响应和迁移结论以人工全帧复核及指标共同判定，脚本状态 `complete_pending_visual_review` 只表示计算完成。
