# EXP-019 执行交接 — Judge → 既有 Exp Worker

2026-10-15 HKT。用户明确纠正：Exp Worker负责执行，Judge负责审核。本轮Judge误承担实现与GPU启动，现全部执行交回既有线程 `01a0ece6-63b6-7f42-8c39-cd3ec39a9c80`。Judge线程 `01a121dc-2c76-7861-8d15-ecfaef383a0f` 不再启动实验。无需新建Worker线程。

## 已完成与现状

- 任务书：根next_plan.md及本目录上级taskbook_v1.md，用户已批准进入该路线。
- 已写独立run.py、CPU_PREFLIGHT.json/log、SOURCE_MANIFEST.json、README、BASELINE_ROADMAP.md；均未commit。run.py是待Worker审阅的草案，不能因Judge编写而免审。
- CPU预检PASS，Judge核对EXP-017 candidate及四fixture/script SHA、EXP-006十项核心冻结代码/Action LoRA摘要。
- `judge/R_APPROVED.json`已签发。R已在GPU0运行，不要重复启动。
- 启动命令：`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-019_v3_mixed_action_validation/run.py --stage R --gpu 0`。
- 启动时PID1283026（接手时用ps重查），root exec session43012。stdout：`submission/experiments/EXP-019_v3_mixed_action_validation/R.log`。输出：`H3-World/outputs/EXP-019_v3_mixed_action_validation/R/`。
- 最近观察R已加载模型，在第1次forward；无结论。首调用可能包含编译。单阶段900秒预算；必要时Worker管理本进程，勿操作其他进程。旧KV已清理，10forward包括两次独立clean commit。
- CPU只读SolarWM EMA hash检查也可能还在运行（exec session99159），结果目标judge/SOLAR_REFERENCE_INVENTORY.json；不涉及GPU。

## 接手职责

1. 先阅读guideline/progress/next_plan与本交接；确认进程/预算/源码SHA/日志。不要改正在运行的run.py，不覆盖已冻结证据；若需修复保留原版本及清单，并交Judge审核。
2. 等待并检查R是否真实完成：C1/C2×sigma1/.5旧新输出，以及50层raw KV逐值一致。固定sigma.5输入是noise*.5探针，不是采样轨迹状态，报告如实说明。审核需留意预算耗时覆盖、实际源导入和cache数量。
3. R完成向Judge提交结果/报告。E和G1/G2尚无marker，不自行跨门。Judge审核后再签发。
4. Worker负责后续实现修复、GPU执行、日志/指标/视频与report.md。Judge负责任务范围、预算、marker、结果审核、next_plan/progress更新。
5. E需审查真实新I0 Qwen head、独立动作句编码、.999 anchor计算、token-refiner spans；实际embedding未来隔离。G需要真实C1建立自身clean KV，两步数相同history/action/noise，不把GT前39帧当新生成。
6. 首scene完成应先报告视觉/动作/结构、全部新增17帧和资源；不为轻微画质反复跑，也不将执行PASS当能力PASS。后续按用户ROI原则。

## 资源

本轮上限166forward/14text/8decode/0train/5400GPU秒，单空闲L40串行，09:00后≤3卡。启动时GPU0–3空闲，4–7有他人任务，接手时重新检查。无新增训练或SolarWM GPU授权；用户批准的总体路线按任务分段具体化。
