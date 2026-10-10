# EXP-012/v1 P0 Worker 报告：冻结 AF8 的共同历史场景迁移准备

2026-10-11 06:54 HKT。**P0 CPU 准备完成，未启动模型/GPU 推理；工业与村落 GPU 阶段均待 Judge 分别签发 marker。** 本任务只检验 AF2 step32 在 EXP-011 已验收的 FM8 首窗历史上生成 C2，不能把旧停车场 AF3 结果当成本轮质量证据。

## 已冻结来源和实际核查

[source manifest](source_manifest.json) 锁定34项来源，SHA-256 为 `95bbdf91608b6b115a09b6b98e3953b99e53d2c59bc6dabd9cef307d7d7d5f7d`；[code manifest](code_manifest.json) 锁定11项任务代码/配置与 CPU audit，SHA-256 为 `4a8e4646ce9e543f3cba14a1765f0cb91e282ce9274be035057e44e3625db1e4`。两者是后续 marker 的绑定依据，不修改 EXP-007、EXP-011 或冻结 runtime。

CPU 已逐项加载并核对唯一 AF2 step32 配套权重与元数据：QKV rank8、blocks42–49、scale0.125、target-time gate0.25、`noise-clean` velocity、相同 step32/任务/协议/精度及 trainer state 中的两个权重 SHA。QKV SHA 为 `07c8e5e68d1c947e60217e326ca8dd4c00222a555542b286dfc03514e27e916e`；target SHA 为 `ceb7d62324834a5b2be9fd6a47bfe4b5a9d810e49d1bb33889af0fb788940f70`。训练来源仍是有限停车场端点，不能推断对这两张 validation 初图已泛化。

两张 EXP-011 native full37/Single I0 fixture、各自 G1/FM8 的 `first12.pt` 与 `published_39.npy` 均与已验收 row SHA 一致；无需新的 text/image encode、C1 生成或 FM8 对照推理。工业和村落 fixture SHA 分别为 `9f404c910c31e68b319cf6029d7d59fa48229d9db14df395a2890cc32057107d` 与 `eb7590213fb7f3971f3e6d345bce185ab8f9b6fba757f43a2123b65631478775`。[CPU audit](P0_CPU_AUDIT.json)记录每个 endpoint/RGB/noise/position 哈希及9点 native8 sigma 网格。检查确认 AA/AD 在当前 C2 只改 action spans 12–16；可见 action rows 为 C1 12 / C2 17；C1 的 Single I0 与前12个 video 的**原生全局位置值**在 C2 视图中一致。

一个值得记录的审计细节：C1→C2 可见裁剪会插入5个 action spans，因而 packed sequence 的物理 `action_video_start` 从1462移到1512。不能把两个 `img_position_ids` 数组按相同物理偏移直接逐值比较；正确对照是各自 video 起点对应的 I0 与前12个 video rows。此项已在 CPU 审计中通过。它是预期的布局变化，不是模型修复或生成能力证据。

冻结 runner 复用 EXP-007 AF3 的 `install_adapters` / `install_anyflow` / `interval_student` / `finite_map_step` 路径，并保留 EXP-011 的 native fixture、Single I0、current-prefix feedback、strict causal / Global RoPE 与 CPU raw KV。每场景先用**AF 自身权重**对共同 FM8 C1 做一次 `sigma=target_sigma=0` 且 `no_grad` 的 clean commit，然后在同一 AF cache 与 C2 初噪声上生成 AA/AD。每支8次 finite-map，对应 native8 邻接 `(sigma, target_sigma)`；旧39 RGB 严格保持不变。不会读取 FM8 hidden KV，也不会重跑 FM8 baseline。

## 预算、闸门和 CPU 测试

预定每场景 **1 commit + 16 sampling =17 denoiser forward、2 decode**；两场景总 **34 forward、4 decode、0训练/编码**，总 GPU 秒≤1260、单卡 allocated≤44GiB、磁盘可用≥60GiB、09:00 HKT 绝对截止。`Ledger` 在每次昂贵调用之前落盘预记账，累计预算不因失败重置；独立输出目录存在则拒绝覆盖，SIGALRM 提供阶段运行中的绝对时间停止。实际 GPU 必须单独检查当前空闲并由 `judge/G1_<scene>_APPROVED.json` 绑定 config/source/code hash；没有 marker 时 runner 在加载33B/GPU之前拒绝启动。环境固定 `ABOT_VRAM_RESERVE_GIB=18`、显式离线模型根、offline flags和FP32精度。

实际 CPU 验证命令：

```bash
cd /home/qma/work/GWM/submission/experiments/EXP-012_v3_af8_scene_transfer
/home/qma/work/GWM/H3-World/outputs/2026-10-09-16/submission_acceptance/venv/bin/python preflight.py
/home/qma/work/GWM/H3-World/outputs/2026-10-09-16/submission_acceptance/venv/bin/python test_p0.py
/home/qma/work/GWM/H3-World/outputs/2026-10-09-16/submission_acceptance/venv/bin/python freeze_code.py
```

结果：真实导入链、两图来源/位置/动作、配对权重、native8网格均 **CPU PASS**；合成预算封顶、finite-map更新符号、缺 marker 的0 GPU拒绝均 **PASS**；`py_compile` PASS。CPU audit JSON SHA-256 `528154464c0a0ff0c6c00cbebb036173896dd3c8988a28548ff46b158499298b`。CPU 测试不代表33B权重安装后的数值或视频质量成功，下一步需 Judge 审核并按场景放行 GPU。

## GPU 阶段待审核命令格式

下面只是后续获批时的命令模板，**不是当前授权**；先工业，完成并目视复核后再决定村落。物理 GPU 编号与 `CUDA_VISIBLE_DEVICES` 必须相同：

```bash
CUDA_VISIBLE_DEVICES=0 ABOT_VRAM_RESERVE_GIB=18 \
DIFFSYNTH_MODEL_BASE_PATH=/home/lpeng/code/mq_PubDataset/GWM/H3-World/DiffSynth-Studio-h3-v2/models \
DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/qma/work/GWM/H3-World/outputs/2026-10-09-16/submission_acceptance/venv/bin/python \
run_exp012.py --scene industrial --gpu 0
```

输出根计划为 `H3-World/outputs/EXP-012_v3_af8_scene_transfer/`；发布只复制小日志、指标、代表视频，raw cache/latents留本机。工业若发生持续结构失败，保留并报告负结果，不自动切换权重、种子或调参。
