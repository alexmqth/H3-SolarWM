# EXP-021/v1 · 普通 FM 训练入口与数据计划（仅草案）

本阶段没有启动 33B 模型、GPU 编码、反向或 optimizer update。目标是把未来一次**可度量的 V3 普通 FM 适配**定义清楚；并不预判它会修复动作和画质。

## 数据和拆分

`DATA_MANIFEST.json` 固定 32 条训练候选源窗口（4 个 train episodes，各 8 条：4 constant-active + 4 switch）与 8 条开发集候选（2 个曾观察的 validation episodes，各 4 条）。每条来自 1800 帧、30 fps 的 ABot 源；选取 56 RGB 帧/24 fps，对应 17 latent 和 `[12,5]` 分块。同 episode 被选窗口的源跨度不重叠，训练和开发 episode 不交叉。选窗依据源动作注释与非重叠约束，不依据生成结果。

**只有 4 条训练 56 帧视频已物化并核对动作矩阵**；28 条训练、8 条开发候选还需实际抽帧、解码、人工/自动画质核查和 VAE 编码。既有 39 帧 clips 不能当 56 帧训练例。现有 2 个开发 episodes 曾用于研究，不能称独立盲测；如要 6–8 个 episode 的盲测，需另引入 4–6 个全新 episodes。物化/编码需要另行核算 CPU/GPU、磁盘和时间，不能算作零成本。

## 模型协议

以冻结 V3 Original H3 + released step-10000 Action LoRA 为基础。保持 Single I0、native time、Global RoPE、own-action/current-prefix feedback、严格 chunk-causal 与 clean raw KV、`12+5` native action/latent 对齐。Released Action LoRA 与 base 冻结；**新建零增量初始化**的 causal adapter。首个 profile 候选是 tail8 QKV rank8，理论新增可训练参数 `8×8×(5376+21504)=1,720,320`；rank/范围须 profile 后批准，不能借旧 AnyFlow/AF32 checkpoint 充当 FM 初始化。

普通 FM 的唯一目标：`z_sigma=(1-sigma)z_clean+sigma*epsilon`，`u=epsilon-z_clean`，优化当前块的 `MSE(v_theta(z_sigma, history_clean, action, sigma), u)`。H3 video timestep 为 `1000*sigma`；不传 AnyFlow target time，不加 finite-map、DMD 或 teacher delta。历史块从 GT clean latent 按各自 native 位置顺序无梯度 commit；当前块读取完整 past KV，但 forward/backward 保持可微。首块训练和后块训练都要计入预算；不能只对当前块做几何正确性假设。

旧 `EXP-002/interval_cached.py` 和 `EXP-005/interval_sw.py` 明确拒绝 `torch.is_grad_enabled()`，是推理入口。旧 `train_stage1_anyflow.py --objective fm` 使用 5-latent/dual-anchor 的旧 real loader，和上述 V3 条件及固定 56 帧动作布局不兼容。**未来需在 EXP-021 后续授权版本中独立实现可微 wrapper**：复用 `visible_inputs` 对 action 与 video 物理截断、`slice_packed` 的全局 RoPE、`ChunkAttention` 与 `current_prefix_feedback`；必须做到 clean commit 无梯度、当前块读 cache 的激活 checkpoint/offload、严格冻结 released LoRA、正确计入训练态显存。不能修改冻结入口来绕过 guard。

`fm_contract.py` 是可单测的数学/接口草案，`test_cpu_contract.py` 使用微型模型验证 FM 符号、σ 端点、未来动作裁剪、history 梯度边界、adapter 更新、optimizer 与 RNG 保存恢复。它**没有**实现上述真实 33B wrapper，也不能证明真实模型能反向或生成正确视频。

## 后续分阶段验证（需 Judge 另行放行）

1. 先物化并核查候选视频；将真实 56 帧编码为 17 latent，并确认前 12 latent 与同源 39 帧编码前缀一致。记录源 SHA、抽帧索引、VAE/conditioner 调用和失败成本。
2. 在固定一条训练 clip 上，做 checkpoint0 完整加载、GT clean history commit、**1–2 次**当前块 forward/backward/update、保存与重载的 profile；测 forward/backward wall、peak allocated/reserved、CPU KV、host RAM、checkpoint 体积及梯度只落在新 adapter。任何 OOM、非有限梯度、cache 读写不一致或未来动作泄漏立即停止。
3. 依据 profile 的实测而非旧推理速度提出训练 updates 上限。保留 checkpoint0 与每阶段 checkpoint，记录 data episode、chunk、sigma、loss、更新次数、optimizer/RNG 元数据；训练/开发严格 episode 隔离。
4. 匹配 NFE、seed、初图和动作，分别在 GT-history 与 self-generated-history、constant 与 switching 案例比较 checkpoint0/训练后。局部画面、动作方向、边界和整段时间均需人工检查；loss 下降不等于能力通过。开发集只有 2 个已观察 episodes，外推结论保持受限。

停止条件：来源/动作布局不一致、训练/开发串集、可微 wrapper 与冻结 V3 前向不等价、base/Action LoRA 有梯度、显存或预算越界、局部视觉明显恶化。未达基本局部生成能力，不扩到 AnyFlow/DMD。
