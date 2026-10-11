# EXP-014/v1 P0 Worker 报告：四张训练初图的冻结教师目标入口

2026-10-11 HKT。`phase=P0` · `status=complete_pending_judge`。本阶段只做 CPU 来源冻结、代码适配和预检：**0 GPU 调用、0 文本/VAE编码、0 denoiser forward、0 decode、0训练更新**。当前没有 P1/T1/T2 marker，未启动任何 33B 模型。

## 固定输入与协议

严格读取 [EXP-013 候选 manifest](../EXP-013_v3_multiscene_data_plan/candidate_manifest.json)，SHA-256 为 `b6af2b0208a5dcf83c02f76fa826e5a03ba52fa9bd83de357f9474b5efb257b6`。四张 train PNG 顺序固定为 `43866101/A_1245`、`7199292c/A_1670`、`9dc2e588/A_1410`、`b784d995/A_1635`；[source manifest](source_manifest.json)逐一核对完整 episode/clip ID、PNG SHA、832×480尺寸、静态 caption 与 clips/annotations/preparation 记录。两个 validation episode 排除。原录屏 A 片段中存在 A+S+L、A+S、A+S+J、纯A；教师 AA/AD 使用显式合成 A/D，而非真实录屏后续标签。

复用 EXP-011 已验收的 native Single I0/full37、12+5 latent、own-action/current-prefix、Global RoPE、CPU raw KV 协议；旧 `encoded/*.pt` 的 Dual Anchor 数据不被读取。四图 P1 预定 12 text/4 image encode；教师每图 C1 A 30步 + sigma0 clean commit + C2 AA/AD 各30步 = **91 denoiser forward / 3 decode**，四图总364forward/12decode，训练更新0。实际加载与推理质量在此阶段尚未验证。

## 实现与资源保护

本目录新建独立 [编码入口](encode_native.py)、[教师入口](run_teacher.py)、[来源核查](source_audit.py)、[P1 fixture 审计](audit_fixtures.py)、[资源/marker/账本](common.py)、[冻结配置](config.json)和 [代码清单](code_manifest.json)；EXP-011 原有代码、结果与 checkpoint 未修改。`common.setup_paths()` 明确设置冻结 DiffSynth 模型目录 `DIFFSYNTH_MODEL_BASE_PATH`，针对 EXP-011 曾发生的离线模型根目录错误。CPU import preflight 实际解析的 editable DiffSynth 位于旧 `2026-10-09-04` 快照，但关键 DiT 与 pipeline 文件与本轮冻结 `2026-10-09-22` 对应文件 SHA 完全相同；冻结清单另外检查后者路径。T1 日志的模型文件也解析到旧快照，但 [29个 `.safetensors` 的 inode 审计](model_alias_audit.json)确认两路径为同一设备上的相同硬链接，总计144,016,376,436 bytes，并非另一个权重版本。若实际 GPU 运行时模块或模型文件身份发生差异应停止，而不能仅凭模块名继续。

P1 使用独立账本；T1/T2 每 scene 使用独立锁和账本，可在 T1 放行、T2 marker 到位后把三图分配到不同空闲卡，输出文件按 scene 隔离。所有昂贵调用**先 reserve 写入账本再执行**；即使调用失败也计费。每 scene 硬上限90 sampling/1 commit/3 decode/1350 GPU秒；P1 为12 text/4 image/600 GPU秒；四 scene 乘法给出5400 teacher GPU秒总上限。剩余磁盘≥60 GiB、相对首次启动增长<40 GiB、allocated 峰≤44 GiB/卡；08:40 HKT后不启动新 teacher scene，09:00后拒绝继续。GPU3/4 按任务书保留给他人，不被本入口使用。P1/T1/T2 需分别由 Judge 提供绑定 config/source/code SHA 且列明 scene 的 marker；缺 marker 时在任何 CUDA/33B 加载前拒绝。

教师同一 scene 的 C1/C2 在同一进程顺序执行，`run_g1` 返回后显式 `gc.collect()` 和 `torch.cuda.empty_cache()`，记录 C2 加载前残余 allocated；若仍≥8 GiB 就拒绝第二次加载，防止同时留住两份33B对象。这个生命周期检查不改变任何模型输入或采样数值；实际残余显存只能在 T1 GPU 运行时确认。

生成链固定同一个 teacher C1 latent 和 C2 初噪声，再 AA/AD 分叉；sigma0 commit 的50层 cache 必须只含 index0，分叉过程中 cache 签名不得变化，原39RGB不得被后续分支改写。每图保存 C1/C2 endpoint、39/56帧视频、逐块 JSON、调用与显存/CPU KV 记录以及 [可重建训练目标索引](run_teacher.py) 所写 `target_manifest.json`。后者只标记 `quality_status=pending_judge`，不能把工程输出自动视为合格目标。学生未来使用冻结 teacher C1 latent、按当前 student 权重重新建立自身 KV；本任务没有学生训练。

## P0 实测检查

`source_audit.py` 生成并复核 [source manifest](source_manifest.json)，四图皆 train、PNG/静态文本/来源 hash 一致。`encode_native.py --preflight` 的 CPU import、directed attention 和37-row packed builder 均 PASS。`test_p0.py` 使用先前已验收的 EXP-011 native fixture 做**协议结构**参考：full37初噪声/单390-row I0，stop12/17 物理移除未来 action/video rows，当前 video 只能直读 own action；50层历史 cache index0、预算计数和无 marker 拒绝均 PASS。该测试不冒充四张新图实际编码结果；P1 完成后必须再运行 `audit_fixtures.py` 检查新生成的四个 tensor。

```text
{"task": "EXP-014/v1", "P0_cpu": "PASS", "source_scene_count": 4,
 "native_full37_reference": "PASS", "future_rows_trimmed": true,
 "own_action_only": true, "cache_50_layer_index0": true,
 "budget_rejection": true, "no_marker_rejection": true, "gpu_calls": 0}
```

冻结摘要：`config.json` SHA `0e33ce856a01590f8ce8f8e9ec76a066ee809db1d3fed737f6e31e6c34ddfcd2`；`source_manifest.json` SHA `35a1da00c8bfc5b2cb73ba05364056164a75187e5422af9262524e1b56a95ebf`；`code_manifest.json` SHA `5afe075f4db7e5b6b107ffabbfcd1feb408dbccd173649b6548c74f876173331`。代码清单覆盖本入口和冻结 runtime 共26文件，不复制33B权重；released Action LoRA SHA 由 EXP-006 来源清单逐次核对。P0 的确切 CPU 命令和待 marker 的 GPU 命令见 [README](README.md)。
