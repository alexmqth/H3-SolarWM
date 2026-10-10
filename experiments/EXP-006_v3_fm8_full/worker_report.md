# EXP-006/v1 Worker 执行报告

`task_id=EXP-006` · `plan_version=1` · `worker_status=complete` · `judge_acceptance=accepted_limited_feasibility`（Judge 原始意见见 `judge/final_review.json`）。

使用 [冻结 runner](run_fm8.py) 与 [配置](config.json)，CPU 预检通过后，在 GPU0 依序运行 `--stage first`、`--stage second --path AA`、`--stage second --path AD`、`--stage third --path AA`、`--stage third --path AD`。每阶段实际命令均为 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-006_v3_fm8_full/run_fm8.py --stage <stage> [--path <AA|AD>] --gpu 0`，对应 stdout 保存为输出目录的 `first_run.log`、`AA_second.log`、`AD_second.log`、`AA_third.log`、`AD_third.log`。首块启动约 2026-10-11 01:41:46 HKT；全部 GPU 阶段结束约 01:52:44 HKT。

输入来自冻结停车场 seed13 A/D conditioning、原始 noise/audio/Single I0 与 released action LoRA，精确文件 SHA、runner SHA `e56437a6bda08ae1f02687c05a3a1df4bf852a718c621c7a011bb9bab8bd27cb` 见 [source manifest](source_manifest.json) 及五份原始 chunk JSON。首窗空 cache，逐块 8 个 native sigma；首/第二块各做 sigma0 clean commit，第三块无需 commit。原始 noise、历史 latent、已发布 RGB 和只读 KV 的 hash/不变性均由 runner 核查；每层 50 层 raw KV，首块每层 4680 token，续块后每层增加 1950 token。未来动作/视频物理裁剪到当前 stop。

最终计数为 43 forwards、5 VAE、零 backward/更新；累计 GPU 占用 492.856 秒，低于任务书 0.75 GPUh；最高 allocated 26.061 GiB，peak RSS 在约 80.6 GiB。各块细节见 [worker_metrics.json](worker_metrics.json)，原账本与中间大文件在 `H3-World/outputs/EXP-006_v3_fm8_full/`。全部 MP4 已完整解码并检查 FPS/帧数/分辨率；两条并排 MP4 也完整解码 73 帧。打包脚本 [finalize.py](finalize.py) 只读取输出并复制小证据。

首块 39 帧人物与停车场连续可辨；AA/AD 两条到 56、73 帧均保持可辨结构。第二块同历史的水平光流 AA +0.846、AD −0.321；第三块各自历史 AA +0.505、AD −0.826。AD 第三块人物尚完整，AA 第三块有持续透明肢体残影，因此只能称普通 FM8 的局部少步可行性，质量仍 PARTIAL。与 EXP-002 30-step 的并排是不同首窗生成 history 的跨协议比较；无法从中单独归因步数或报告公平加速比。没有证明 124 帧、跨 scene/seed 或 AnyFlow 能力。
