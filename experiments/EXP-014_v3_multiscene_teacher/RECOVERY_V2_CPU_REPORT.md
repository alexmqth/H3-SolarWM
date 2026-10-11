# EXP-014/v2 Worker CPU 报告：仅补两条缺失 AD 的恢复入口

后续GPU恢复已按Judge新marker完成，实际结果见[恢复GPU报告](RECOVERY_V2_GPU_REPORT.md)。本报告记录GPU启动前的CPU审计与冻结状态。

2026-10-11 HKT。按照 Judge [v2 恢复任务书](taskbook_v2_recovery.md)，已完成**仅 CPU**的独立恢复入口、来源冻结和预算预检；**GPU 恢复尚未启动，新增 denoiser/decode/训练调用均为0**。原 v1 `run_teacher.py`、`common.py`、配置、原始输出、半途 AD 行及未关闭账本均未修改。新入口只允许 `s1_7199292c` 与 `s3_b784d995` 各自补做一次原 30-step AD，不重算 C1、AA 或 clean commit。

## 实现与冻结文件

| 文件 | SHA-256 | 用途 |
|---|---|---|
| [recovery_config.json](recovery_config.json) | `26a1fc06d97996be61ac2aaced6b0280428f35a0378dd7a94829eef27bb101be` | 两场景、30F/1decode/600秒单图、09:00截止 |
| [recovery_source_manifest.json](recovery_source_manifest.json) | `43d597a1482a5dd1d5d6ba072a41fb13f9065bd2db5f166d86aeae9d2814d066` | v1源码/清单、两场景C1、AA、cache、旧AD行与账本逐文件摘要 |
| [recover_ad.py](recover_ad.py) | `fc9aecc742923560b86c9861b27bd1c64b5fae29bf8c0e7632c5ce30702a1a76` | CPU预检、marker门、独立write-ahead账本和缺失AD生成 |
| [recovery_code_manifest.json](recovery_code_manifest.json) | `b294c0d1a55a81c374a0c1741f50e59127bad1c4102c0211a55ca0bef7c416f4` | 冻结前三个文件；marker需绑定该SHA |

CPU [预检结果](artifacts/recovery_v2_cpu_preflight.json)两图均 `PASS_CPU_ONLY`。逐图重新读取实际6,799,104,000-byte raw KV，确认50层各一条 `index=0`、C1/AA endpoint与公布RGB SHA一致；原 AD 尚无 endpoint/video。重建第12–16个 D action span，核对除了这些 span 外prompt不变，恢复的 AD prompt、native Global position、C2初始噪声 SHA与原中断 AD 行**逐值同协议**；AA/AD 使用同一31点 sigma 网格。原账本 s1 预留78F/AD已持久化17步，s3预留80F/AD已持久化19步。它们只用于来源审计，恢复从 AD 第一步重新计算，因为中间 noisy latent 未保存。

新输出固定到 `H3-World/outputs/EXP-014_v3_multiscene_teacher/recovery_v2/<scene>/`，包含独立账本、AD endpoint、56帧原片、17帧新增片和新 target manifest，**不覆盖**旧 `G2/<scene>/FM30/AD/`。每个场景目录存在就拒绝再次尝试。账本先预留再执行模型调用，单图最多30 sampling、1decode、0commit/0训练、600 GPU秒；两个固定场景且每场仅一次使总新增上限为60F/2decode/1200秒。调用前还检查磁盘下限、物理卡空闲、显存阈值、08:40新任务截止与09:00硬截止。对 SIGTERM 有异常关闭处理；不能以此保证未知外部终止一定留下完整日志。

CPU [预算单元测试](artifacts/recovery_v2_cpu/recovery_v2_ledger_test.log)验证第31次sampling及第2次decode被拒绝、commit/update保持0。无 Judge 恢复 marker 时，[运行入口拒绝测试](artifacts/recovery_v2_cpu/recovery_v2_missing_marker_test.log)预期失败且没有创建恢复输出目录。代码清单单独核验PASS。以上都未加载33B模型到GPU。

## 精确执行命令和下一阶段边界

从 GWM 根目录，CPU预检已执行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venvs/h3world/bin/python submission/experiments/EXP-014_v3_multiscene_teacher/recover_ad.py --preflight
```

**下列命令是待 Judge 新 marker 后才可使用的入口，本轮未执行：**

```bash
EXP014_MODEL_ROOT="$PWD/H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/models"
CUDA_VISIBLE_DEVICES=0 DIFFSYNTH_MODEL_BASE_PATH="$EXP014_MODEL_ROOT" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venvs/h3world/bin/python -u submission/experiments/EXP-014_v3_multiscene_teacher/recover_ad.py \
--run --scene s1_7199292c --gpu 0 \
> H3-World/outputs/EXP-014_v3_multiscene_teacher/recovery_v2_s1.log 2>&1

# 仅在第一图完整退出并审计后，再使用同一个实时空闲GPU0运行s3：
CUDA_VISIBLE_DEVICES=0 DIFFSYNTH_MODEL_BASE_PATH="$EXP014_MODEL_ROOT" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venvs/h3world/bin/python -u submission/experiments/EXP-014_v3_multiscene_teacher/recover_ad.py \
--run --scene s3_b784d995 --gpu 0 \
> H3-World/outputs/EXP-014_v3_multiscene_teacher/recovery_v2_s3.log 2>&1
```

marker 位置为 `judge/RECOVERY_APPROVED.json`，必须包含 `task=EXP-014/v2`、`approved=true`、确切scene白名单及上述 config/source/code manifest SHA。Worker 不创建 marker；Judge 应在审核本报告与 CPU preflight 后签发。即使两图恢复完成，也只代表四图目标的完整性，具体动作与人物质量仍须逐图审视；`s2` 已有的D反转 PARTIAL 不会因恢复其他场景而自动转为合格监督。后续 AnyFlow/DMD 仍不在本任务授权内。
