# EXP-014/v1 P1 Worker 报告：四张 native Single I0 fixture

2026-10-11 HKT。按 Judge 的 [P1 marker](judge/P1_APPROVED.json)，GPU0 完成四张预先冻结 train 初图的 native full37 输入编码。P1 实际 **12 text encoder forward、4 image VAE encode、0 video VAE encode、0 denoiser forward、0 decode、0 backward/update**；账本时间 **110.214 GPU秒 / 600秒**，`torch.cuda.max_memory_allocated` 峰值 **40.578 GiB / 44 GiB**。单次运行成功，无编码重试或覆盖旧 `.pt`。完整原始 [调用账本](artifacts/P1/budget.json)、[结果](artifacts/P1/P1_result.json)、[stdout](artifacts/P1/P1.log)已复制小文件归档，四个22MB左右 fixture 原件保留 `H3-World/outputs/EXP-014_v3_multiscene_teacher/fixtures/`，本目录只记 SHA。

运行命令（GWM 根目录；GPU0 当时空闲45,458 MiB）：

```bash
CUDA_VISIBLE_DEVICES=0 \
DIFFSYNTH_MODEL_BASE_PATH=/home/qma/work/GWM/H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/models \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venvs/h3world/bin/python -u submission/experiments/EXP-014_v3_multiscene_teacher/encode_native.py --encode --gpu 0 \
> H3-World/outputs/EXP-014_v3_multiscene_teacher/P1.log 2>&1
```

| scene | fixture SHA-256 | bytes |
|---|---|---:|
| `s0_43866101` | `a0b6ce46676df740bbe884535341319a2dd363f8640eb3fbbb432f75d18b8e4e` | 22,326,653 |
| `s1_7199292c` | `2965f591b093f1eb4a64b23b4f31afd307766a08bb00123209967bd58e71dd38` | 22,675,005 |
| `s2_9dc2e588` | `28aa37f742f6bdb029fff08d75b9a380e3c77280466a52ff48552c4ab4f8a951` | 22,818,493 |
| `s3_b784d995` | `934570b5dd9d0625d1a97724f71a9f68cc730af3b4d0e2377061dcb0166d6a52` | 22,716,029 |

Worker 的 [四fixture CPU audit](artifacts/P1/P1_cpu_audit.json) 为 PASS：每张 PNG 来源 SHA 正确、一个390-row I0、full37个动作 span，A/D 非动作文本与位置相同；stop12/17 分别物理删去25/20个未来动作 span，当前视频只直读 own action；四图 initial video/audio noise 一致。Judge 独立的 [native 输入审计](judge/P1_INDEPENDENT_AUDIT.json)另外从 seed13/124帧原生 NoiseInitializer 逐值重建 video/audio 噪声，核查 Global 位置保持与 A/D action embeddings 差异，并判 PASS。审计均为 CPU 0 GPU 调用，证据不代表视频动作或视觉已通过。

本轮只生成输入，没有 C1/C2 endpoint 或教师视频。Judge 已另行签发 T1 marker，允许固定 `s0_43866101` 做首图 FM30 C1→AA/AD C2；T2 余图仍未放行。T1 的实际画质、动作和KV会单独报告。
