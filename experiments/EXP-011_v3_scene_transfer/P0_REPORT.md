# EXP-011/v1 P0 CPU 审计与 GPU 阶段准备

执行时间：2026-10-11 05:40–05:52 HKT。**本报告仅覆盖 CPU 设计与预检；没有运行 P1 编码、G1/G2 采样、训练或 VAE decode。**

## 输入来源

| 场景 | sample/clip | PNG SHA-256 | split | 来源 |
| --- | --- | --- | --- | --- |
| industrial | `118eb5d8b75e1b8ac23a4e9ae77af9a9/A_1140` | `5e6bd9a2c137e8df3b86cab99bb1861f7e226b172fb9316be2069d3ba02b2774` | validation | ABot-World-Explorer-500h 视频 30 fps，原片 SHA `100199334fb037922b8e804192592cbf1cbd3090dc3fd248d17c8dc75d3888a8` |
| village | `dfec8ed3237860eba14d67c089ecd041/D_1750` | `961f3492d3a7517a151ffec97da4a2fbeee5460e3ead9d0d91787c29ce2ed880` | validation | 同数据集，原片 SHA `6e4050d896166a9aebee3b62de22b4c7e96da3ede1cf7274c5c9e17c19a38432` |

两张 PNG 都是 832×480，哈希与 `preparation.json` 的 `first_frame_sha256` 一致；`encoded_manifest.json` 与 `preparation.json` 均确认 validation。静态场景文本精确采用各 episode 的 `annotation_pilot.json → caption.scene_static`，并与 `preparation.json` 的 `prompt` 一致；不取 `narrative`。来源细节和完整描述冻结在 [source_manifest.json](source_manifest.json)。旧 encoded manifest 的协议为 `global_retimed_rgb_prefix_last_image_dual_v2`，本轮不加载其 `.pt`。这些是游戏录屏的两例固定场景，不是统计泛化样本。

## 原生输入与信息流审计

冻结运行时的输入链为 `MiniMaxH3Unit_ShapeChecker → NoiseInitializer → KeyframeEncoder → PromptEmbedder → PackedSequenceBuilder`。具体代码在冻结 [H3 pipeline](/home/qma/work/GWM/H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py)；P1 使用该 unit chain，`num_frames=124` 得到 37 个 latent time slices 与 `[1,24,37,30,52]` 初噪声，首帧经 `process_image=True` 生成 **一个** 390×96 anchor。`keyframe_indices=[0]`，`cfg_scale=1`，seed13；两场景从相同 seed 各自重新初始化 video/audio noise。full37 的 packed 序列先构建一次以固定 action/video、audio、Single I0 和全局 RoPE 坐标，`visible_inputs` 在每次调用前物理删除 `stop` 之后的 action/video rows，同时保留可见 rows 原全局位置。

A/D 句子由冻结动作脚本产生：`the man strafes left, camera follows him` 与 `the man strafes right, camera follows him`。CPU tokenizer 预检两者各 10 token，因此同场景可以只替换 action embeddings 而保留一个 packed 位置合同。P1 仍会检查真实 embeddings 的 37 个 span 完全相同，否则停止并交由 Judge 复审。G1 只读 A 前 12 个动作；G2 AA 保留 A，AD 只替换第 12–16 个 action span，未来 rows 不进入 DiT。V3 `interval_cached` 使用 own-action 路由、action feedback、current non-action prefix feedback、native timestep/audio1000、Global RoPE、strict chunk-causal raw KV；G2 的 C1 sigma0 clean commit 后才读自身历史。对同一方法，AA/AD 共用同一首窗和已提交的 50 层历史 KV；不同方法的历史不是同一个状态。

不能在 P0 断言新场景实际 packed、噪声和 action delta 已正确：必须在 P1 新编码后由 [audit_fixtures.py](audit_fixtures.py) 对真实 tensor 逐项验证。P0 只完成代码路径与 CPU 静态/合成检查。新场景的画质、动作方向和多场景迁移仍未知。

## P1 精确 GPU 调用预算

每场景：PromptEmbedder 的 static scene + Single I0 head **1** 次 text encoder forward、相同 A action sentence 经去重 **1** 次、D 句子另外 **1** 次；共 **3 text encoder forwards**。KeyframeEncoder 对单张 PNG 调 `encode_video(process_image=True)` **1** 次；没有视频 VAE encode。两场景合计 **6 text forwards、2 image VAE encodes、0 video VAE encodes、0 denoiser forwards、0 VAE decodes**。实际次数由 text forward hook 和 VAE encode wrapper 在调用前写入账本，数目不符即失败。P1 只加载 text encoder、source VAE 和 processor，不加载 33B DiT。整个 P1 上限 900 GPU 秒；这还是计划预算，不是已测运行时间。

G1/G2 后续预算与任务书一致：G1 76 sampling / 4 decode；G2 152 sampling + 4 commits / 8 decode；推理累计 232 forwards / 12 decode，推理 4500 GPU 秒，连 P1 总 5400 GPU 秒。每次只使用一张真正空闲卡，所有 GPU 秒累计。脚本在每次模型调用前写账、检查 09:00 HKT 截止和 ≥60 GiB 空闲磁盘；显存 allocated 上限 44 GiB，输出存在时不覆盖，无自动重试。

## CPU 验证和待放行步骤

`encode_native.py --preflight`：PASS，确认冻结 directed attention、per-latent packed builder、runtime imports、两输入来源；0 GPU calls。`test_p0.py`：PASS，确认来源/代码哈希、bf16 初噪声到 fp32 比较口径、实际内存 KV 版本变动可检测、P1 调用超额/禁用项拒绝；0 GPU calls。独立 `--encode --gpu 0` 在缺失 `judge/P1_APPROVED.json` 时如预期拒绝，未开始加载 GPU 模型。`py_compile` 与 `git diff --check` 通过。

后续 P1 首次获批启动时，在模型加载前出现字符串路径哈希类型错误，详见 [P1_ATTEMPT1_FAILURE.md](P1_ATTEMPT1_FAILURE.md)。该失败不是 P0 CPU 测试的成功结果；修复后已补充字符串路径回归检查并重新冻结代码，需 Judge 新 marker 方可重试。

P1/G1/G2 各自需要 `judge/` 中相应 marker，绑定 config、source manifest、code manifest 的 SHA；本 Worker 不创建批准文件。P1 若放行，先运行编码，再运行 CPU `audit_fixtures.py` 检查实际 A/D 非动作文本、噪声、camera/video 位置、Single I0、future-action trimming、own-action mask 和 fixture SHA。此后 G1 的四个 39 帧原片需完整目视复核，G2 仅对 Judge 放行的场景/方法执行。
