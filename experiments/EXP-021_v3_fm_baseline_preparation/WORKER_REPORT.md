# EXP-021/v1 Worker 报告：CPU 准备完成，待 Judge 验收

2026-10-15 16:27 HKT。严格执行 [任务书](taskbook_v1.md) 的 CPU 阶段；0 GPU forward/encode/decode、0 训练更新、无新视频、未更动冻结源码或权重。`submission` HEAD 起点为 `04a3aeaa47763862bbe996c2e7ef658531590674`；本轮不执行 Git add/commit/push。当前新增文件约 216 KiB（根报告快照后再核算），远低于 2 GiB。使用 `.venvs/h3world`、`CUDA_VISIBLE_DEVICES=''`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=4`；脚本和测试均为 CPU。首轮操作在 30 分钟内完成。

## 结果与证据

1. [来源清单](DATA_MANIFEST.json) 追溯 EXP-013 的六个 ABot episodes 和源 SHA。4 个 train episodes 各 8 个不重叠的 56 帧候选窗口，合计 32（16 constant-active、16 switch）；2 个**已观察** validation episodes 各 4 个，合计 8（4+4）。训练与开发 episode 隔离；[独立结构审计](MANIFEST_AUDIT.json)再次核查 40 条 ID、69 个源帧跨度、30→24 fps 索引、`[12,5]` action layout、类别与不重叠。来源检查见 [CPU_CHECKS](CPU_CHECKS.json)。
2. 仅 4 条 train `real56.mp4` 已存在，且本次重算的 `raw56.npy` 逐值匹配；**28 train + 8 dev 仍是源窗口候选**，没有抽帧、RGB 视觉核查或 VAE 编码。现有 39 帧 clips 不是本任务的 56 帧样本。只有 2 个开发 episode 且早已观察，缺少 4–6 个新 episode 才可能建立 6–8 场景盲测；目前不能宣称泛化通过。
3. [FM 接口草案](fm_contract.py) 与 [4 个 CPU 单测](test_cpu_contract.py)通过，命令及输出见 [日志](CPU_TEST_LOG.txt)。验证 `zσ=(1−σ)clean+σnoise`、目标 `noise−clean`、当前块前的未来 action 裁剪、clean history 无梯度、微型冻结 base/可训 adapter 更新、optimizer/RNG 保存及协议不匹配时拒绝载入。测试使用微型 CPU 模型，**不验证真实 H3 33B 可微反向、attention/KV 数值、视觉或动作**。
4. [训练计划](TRAINING_PLAN.md)说明关键阻塞：冻结 `interval_cached.py` 和 `interval_sw.py` 在 grad-enabled 时抛错；旧 `train_stage1_anyflow.py --objective fm` 的数据/anchor/chunk 协议与 V3 不符。下一阶段需独立可微 wrapper、真实条件与编码、完整 profile，不能沿用旧 AF32 checkpoint 冒充纯 FM。tail8 QKV rank8 的 1,720,320 可训参数只是候选，不是已实现训练配置。
5. [GPU 预算草案](GPU_BUDGET_DRAFT.md)先建议 1–2 个完整更新的实测 profile，再决定可承受的训练规模；编码、失败调用、验证与 checkpoint 均须记账。用户夜间最多 8 卡、次日 09:00 后最多 3 卡的限制已记录；**本任务 GPU 授权仍为 0**。[SolarWM 准备页](SOLAR_STAGE2_REFERENCE.md)核对官方 Stage2-fix 本地权重与 8 卡/SP4/158f 配置，但本地未发现必需的 preencoded test index、silence latents 和 encoder contract，不将其视作已复现。

## 实际执行命令与资源

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-021_v3_fm_baseline_preparation/prepare_data.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-021_v3_fm_baseline_preparation/audit_manifest.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venvs/h3world/bin/python -m unittest discover \
  -s submission/experiments/EXP-021_v3_fm_baseline_preparation -p test_cpu_contract.py -v
```

`prepare_data.py` 实测约 2.17 秒（见 CPU_CHECKS），独立结构审计通过，4/4 CPU 测试通过。`prepare_data.py` 与 `audit_manifest.py` 拒绝覆盖产物，保留本次测量和结果。因未启动 33B，训练耗时、峰值显存、模型效果均为 **NOT_TESTED**；不能依据旧推理成本填数。

## 建议交接

Judge 可先验收来源与接口草案。若放行下一阶段，优先补新窗口物化/质量检查和**可微 V3 wrapper 的 1–2 update profile**，按 profile 再确定训练上限。SolarWM 官方标准相机推理需单列任务并先补足官方数据依赖。夜间卡数增加不能替代冻结任务书的 0 GPU 限制。
