# EXP-014 — 四张训练初图的原生 V3 FM30 教师目标

P0 来源冻结、P1 四图原生输入和 T1 首图教师目标均已完成并经 Judge 分阶段验收。T2 三图中，`s2_9dc2e588` 完成 AA/AD，协议PASS但D反转未证实；`s1_7199292c`、`s3_b784d995` 的AD进程曾收到SIGTERM，原始失败记录保留。v2按新marker**仅补齐这两条AD**，两图均协议PASS；`s1` D反向较清楚但A持续性弱，`s3` 两路几乎同向，动作目标不通过。详见[恢复GPU报告与并排视频](RECOVERY_V2_GPU_REPORT.md)、[T2中断报告](T2_WORKER_REPORT.md)和[v2恢复CPU准备](RECOVERY_V2_CPU_REPORT.md)。此前阶段见[P0](P0_REPORT.md)、[P1](P1_WORKER_REPORT.md)、[T1](T1_WORKER_REPORT.md)报告。每个 GPU 阶段都有 Judge 单独签发的 marker；新训练更新始终为0。预算与停止条件见[v1任务书](taskbook_v1.md)及[v2恢复任务书](taskbook_v2_recovery.md)。

固定场景来自 [EXP-013 候选清单](../EXP-013_v3_multiscene_data_plan/candidate_manifest.json)，四图按 `s0_43866101`、`s1_7199292c`、`s2_9dc2e588`、`s3_b784d995` 排序。[config.json](config.json)、[source_manifest.json](source_manifest.json)、[code_manifest.json](code_manifest.json)绑定输入、源码及冻结 runtime；模型大权重和 endpoint 只保存在 `H3-World/outputs/EXP-014_v3_multiscene_teacher/`。

快速审阅四图可用性见[Worker目标质量清单](WORKER_TARGET_ASSESSMENT.json)，完整预留/已确认调用与中断资源上下界见[Worker累计账本](WORKER_CUMULATIVE_BUDGET.json)。两者待Judge最终审查；原始逐调用账本、视频和阶段报告仍是事实来源。

P0 CPU 命令（已通过）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-014_v3_multiscene_teacher/source_audit.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-014_v3_multiscene_teacher/encode_native.py --preflight
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-014_v3_multiscene_teacher/test_p0.py
```

`freeze_code.py` 在所有代码/config完成后运行，生成当前冻结清单；任何被冻结文件再改动都会使原 marker 失效，必须重冻并由 Judge 重新放行。

以下是**待 marker 才能运行**的完整入口；所有命令从 GWM 根目录执行，选用实时空闲、非 GPU3/4 的物理卡。`common.setup_paths()` 也会在程序内设置相同的冻结模型根目录，防止独立进程错找离线模型。

```bash
mkdir -p H3-World/outputs/EXP-014_v3_multiscene_teacher
EXP014_MODEL_ROOT="$PWD/H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/models"

# Judge 的 P1_APPROVED.json 存在并通过哈希检查后：
CUDA_VISIBLE_DEVICES=0 DIFFSYNTH_MODEL_BASE_PATH="$EXP014_MODEL_ROOT" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venvs/h3world/bin/python -u submission/experiments/EXP-014_v3_multiscene_teacher/encode_native.py --encode --gpu 0 \
> H3-World/outputs/EXP-014_v3_multiscene_teacher/P1.log 2>&1

# P1 完成后只用 CPU 验证四 fixture；同时需 Judge 独立审计：
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venvs/h3world/bin/python submission/experiments/EXP-014_v3_multiscene_teacher/audit_fixtures.py

# Judge 的 T1_APPROVED.json 包含 scenes=["s0_43866101"] 后：
CUDA_VISIBLE_DEVICES=0 DIFFSYNTH_MODEL_BASE_PATH="$EXP014_MODEL_ROOT" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venvs/h3world/bin/python -u submission/experiments/EXP-014_v3_multiscene_teacher/run_teacher.py \
--stage T1 --scene s0_43866101 --gpu 0 \
> H3-World/outputs/EXP-014_v3_multiscene_teacher/T1_s0.log 2>&1
```

T2 marker 只能在 T1 的完整视频/动作/结构验收后发布。T2 三场景可用三张当时空闲卡并行，分别用 `--stage T2 --scene s1_7199292c`、`s2_9dc2e588`、`s3_b784d995`；每条命令的 `CUDA_VISIBLE_DEVICES`、`--gpu` 和日志文件必须各自对应，不能共用一个 GPU 设备或输出路径。08:40 HKT 后不得启动新的 teacher scene，09:00 HKT 后项目最多3卡且本任务入口停止。

marker 至少包含 `task=EXP-014/v1`、对应 `stage`、`approved=true`、当前 config/source/code manifest SHA；T1/T2 还要包含确切 `scenes` 白名单。Worker 不创建 marker。代码先验拒绝不存在或不匹配的 marker，账本在每次昂贵调用前保存；若结果失败，保留原目录与日志而不覆盖重跑。
