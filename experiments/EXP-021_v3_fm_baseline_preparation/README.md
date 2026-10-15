# EXP-021/v1：普通 FM 因果适配与真实数据准备

**Worker CPU 阶段已完成，待 Judge 验收。** 本轮无 GPU 调用、训练、VAE 编码或视频生成；不能从此目录推断 FM 模型能力改善。[冻结任务书](taskbook_v1.md) · [Worker 报告](WORKER_REPORT.md)。旧根报告在 [历史快照](REPORT_BEFORE_EXP021.md)，本轮只改本目录及根 `report.md`。

| 交付 | 内容与状态 |
| --- | --- |
| [数据清单](DATA_MANIFEST.json) / [来源检查](CPU_CHECKS.json) | 32 条 train 和 8 条 dev 的 56 帧**候选源窗口**；仅 4 条 train 已物化，其余待解码、VAE 编码、视觉核查 |
| [独立结构审计](MANIFEST_AUDIT.json) | episode 隔离、源跨度不重叠、动作类别和 `[12,5]` 布局通过；不是 RGB/33B 审计 |
| [FM 接口草案](fm_contract.py) / [CPU 测试](test_cpu_contract.py) | FM 符号、未来动作截断、梯度边界、adapter 更新和 checkpoint/RNG 恢复；[测试日志](CPU_TEST_LOG.txt) 4/4 通过 |
| [训练计划](TRAINING_PLAN.md) / [GPU 预算草案](GPU_BUDGET_DRAFT.md) | 可微 V3 wrapper 的实现缺口、profile、停机条件；草案不构成授权 |
| [SolarWM 官方 Stage2 准备页](SOLAR_STAGE2_REFERENCE.md) | 权重已在本地；官方预编码 test index/support 文件未发现，推理未运行 |

CPU 复核命令（在 `/home/qma/work/GWM`，使用项目环境）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venvs/h3world/bin/python -m unittest discover \
  -s submission/experiments/EXP-021_v3_fm_baseline_preparation \
  -p test_cpu_contract.py -v
```

`prepare_data.py`、`audit_manifest.py` 为一次性、拒绝覆盖既有 JSON 的来源审计脚本，产物已保存；如要重建，先保存原件并在新目录运行。无视频是本轮任务范围所致。

## Judge验收

CPU准备已accepted收口，真实H3训练尚未验证。[审核](judge/FINAL_REVIEW.md)。后续实际实现与有限GPU profile归EXP-022。
