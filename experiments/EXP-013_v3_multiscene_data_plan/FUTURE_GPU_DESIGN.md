# EXP-013 后续 GPU 阶段草案（本任务不执行）

此文档是独立任务的预算建议，不是运行许可。当前 EXP-013 只做 CPU 审计，GPU 调用和训练更新均为 0。用户说明 09:00 HKT 前最多可用 8 张空闲卡，之后项目最多用 3 张；实际调度仍以新的 Judge 任务书和当时的空闲情况为准，不因有空卡而增加样本或更新数。

## 输入与可用性

| 材料 | 当前状态 | 后续用途与限制 |
|---|---|---|
| 6 个 ABot 原始 episode、24 个 39 帧 clip、17 列动作 | 本轮可直接审计 | 16 train clip 来自 4 个 episode；8 validation clip 来自 2 个已观察的固定回归 episode。同一 episode 的 clip 不独立。|
| 4 张确定性 train 初图 | `candidate_manifest.json` 已冻结 | 每个 train episode 取 `target=A` 中最小 `src_start`；只把 PNG 和静态 caption 用作新模型输入。录屏动作有 A+S、A+S+L/J，不能作为纯 A GT。|
| 旧 `encoded/*.pt` | Dual Anchor 协议 | 不能喂给 V3 native Single I0；不可把旧 `.pt` 改名或复制当新 fixture。|
| EXP-011 native fixture 代码和两验证初图 | 已验证编码链 | 复用源码和协议，针对 train 初图重新编码 full37；旧 validation fixture 只作固定回归，不混入训练。|
| 39RGB/12latent clip | 只有 C1 | 没有现成 56RGB C1+C2 GT。若未来要用真实 C2，需从源 MP4 按 `window_offsets(56)` 重新取帧，并用同一源 annotations 构造 56×17 动作；本任务没有做此重切或 VAE encode。四张候选起点均满足 1800 源帧边界。|
| 冻结 V3 FM30 C1/C2 AA/AD 教师目标 | 尚不存在于四 train 初图 | 必须另立 GPU 任务生成；它是模型生成的反事实伪标签，不是录屏 GT。|

EXP-007 的停车场训练输入是冻结的 V3 generated C1/C2 AA/AD；已有清单不提供可与这六个 ABot episode 做严格哈希或 ID 匹配的原始停车场 episode。因此不能证明其源 episode 独立性。将停车场只当已观察的回归参考，不作为本轮盲测或统计独立性的证据。

## 阶段 1：重新编码 native Single I0（需新任务）

输入严格采用四张候选 PNG 与 `caption.scene_static`；`num_frames=124` 只用于建立 full37 原生 packed 位置和同 seed13 的 video/audio 初噪声。只放置一个经 `process_image=True` 编码的 I0 image anchor，不做视频 VAE encode。对每张图编码 static/A/D 三类文本，即 **3K=12 text forward、K=4 image VAE encode**。物理删除当前 stop 之后的 action/video rows；校验 A/D 只有对应动作 embeddings 不同、Single I0 只有一个 anchor、full37 的 37 个 span 和全局位置一致。预期每 fixture 约 22 MB，四个约 88 MB。EXP-011 两图 P1 实测 6 text+2 image encode、70.565 GPU 秒（包括两次加载前失败），一次成功运行 61.897 秒、allocated 峰值 40.576 GiB；四图估计约 2–4 分钟 GPU wall，建议阶段上限 10 分钟，显存上限 44 GiB。旧编码绝对不能复用。

**门 1：** 四 fixture 来源 SHA、Single I0、A/D 位置和未来裁剪均通过 CPU 审计，否则不启动教师生成；失败保留，不换图。

## 阶段 2：冻结 V3 FM30 教师目标（需新任务）

每初图固定 A 动作生成 C1 12 latent/39 RGB；对自身生成的 C1 做一次 sigma0 clean commit，固定该 generated history，再分叉 AA 和 AD 的 C2 5 latent/17 新 RGB。V3 Original H3 + released Action LoRA 冻结；strict chunk causal/current-prefix/Global RoPE/native shift2.22/CPU raw KV 不变。无 GT reset。每图 **30 + 1 + 2×30 = 91 denoiser forward、3 VAE decode**；K=4 总计 **364 forward、12 decode**。不永久保存约 6.33 GiB/历史的 raw KV，存 C1/C2 endpoint、完整来源/噪声/调用账本及视频。

EXP-011 两图四 FM30/FM8 配置总计 232 forward、12 decode、1781.710 GPU 秒、allocated 峰值约 26.12 GiB。按实测整体成本外推，364 forward 约 2800 秒；考虑不同场景/重载，建议**单卡 GPU 总预算 1.5 小时**和明确的每次调用账本，而不是用 NFE 比值声称端到端速度。四图可按 episode 独立排队，不必为了并行占满八卡。首图生成后先看完整 39+17 帧与 A/D 分叉；严重持续结构失败时保留失败样本并暂停，由 Judge 决定是否继续余下图，不自动换 seed/图/动作。

**门 2：** 来源、缓存、旧 RGB 不变、A/D 同 C1 状态、视频完整性通过；教师目标结构及动作是否足以作伪标签由 Judge 人工验收。若教师本身失去 A/D 或人物结构，不能把它当正确训练目标。

## 阶段 3：独立多场景 AnyFlow student（需新任务）

建议从冻结 V3 权重重新初始化一个**独立**的 target-time gate0.25 + tail8 rank8 QKV student；不从 EXP-007/AF2 step32 继续，以免把单停车场训练效果与数据变化混淆。训练的 C1 clean latent 和 AA/AD C2 endpoint 来自**一次生成后冻结的 V3 FM30 teacher**；每次更新不重新采样 student C1。沿用 EXP-007 的 noise-clean velocity、shift2.22、epsilon5 和逻辑 batch4：2 个 diagonal/FM、1 个 endpoint、1 个 general finite interval。当前点 `z_t=(1-t)z_clean+tε`，同状态上求 `v(z_t,t,r)`；有限映射为 `z_r=z_t+(r-t)v`。目标残差含 `v+(t-r)∂_t v-(ε-z_clean)`；`r=t` 是 FM/diagonal 约束，不等于少步视频已通过。`t/r` 从官方式采样后 shift；每样本三次 detached 目标调用加一次可微调用，先按 logical batch 的 FM 基准作自适应缩放，再乘时间权重。每次更新前把同一冻结 teacher C1 latent 用**当前 student**权重做 sigma0 clean commit，以重建各场景自身 KV；不能跨权重、跨场景或师生共用隐藏 KV。

预定物理 batch1、logical batch4、每个 update 每个 train episode 一项。8-update 循环可设 episode 序号 `e=0..3`、update 序号 `u`：action=`A` 若 `(u+e)%2=0`，否则 `D`；sample type=`floor(u/2)+e (mod 4)`，使每个 episode 在 8 update 内分别见到 A/D×4 类目标一次。建议总 **32 update** 固定最终 checkpoint，不用 validation 选 step。每 update 四个 episode 各一次 clean C1 prefill +16 个 finite-map forward，即 **20 forward、4 backward**；总 **640 forward、128 backward、32 update**。顺序处理四 cache，peak CPU raw KV 约 6.33 GiB 而非同时持有四份。EXP-007 单场景 31 update 实测 527 forward/124 backward、4287.861 GPU 秒、allocated 峰 26.767 GiB；额外三次 prefill/update 估计增加约 750 GPU 秒。建议总上限 **2.5 单卡 GPU 小时**、allocated ≤44 GiB。最终配套 checkpoint 参考 EXP-007 step32 约 201 MB；只保留固定终点与必要中断恢复点，不保存每步 KV/全量 33B。此预算是外推，不保证质量改善。

存储规划：四个新 native fixture 约 88 MB；教师各场景 C1/C2 endpoint、MP4 和日志拟上限 400 MB；单个配套 student 终点参考约 201 MB；固定评估视频/小型审计拟上限 300 MB。各阶段不落盘 6.33 GiB/场景的 raw KV、不复制33B底座；建议先按新增≤1 GiB、剩余空间≥60 GiB 设停线。实际落盘逐文件计账；若决定保存额外中断 checkpoint 或 KV，必须另修预算。

**门 3：** 更新数、finite loss、梯度、base 冻结、KV 权重版本、optimizer/RNG/checkpoint 配对全部成立，才进入视频评估。Loss 下降只是训练链路证据，不能替代动作/画质验收。NaN、基础权重改变、协议错配、OOM、超过预算立即停止；无自动重试或扩大 update。

## 阶段 4：固定回归视频（需另放行）

两 validation episode（工业、村落）和停车场是**已观察回归集**。冻结 scene/seed13/动作/噪声/分辨率/24FPS/native 8-step sigma；FM8 和 AF8 各自建立与自身权重相符的 KV。普通 FM8 的两验证场景已有 EXP-011 的39帧 C1 和 AA/AD 56帧，停车场已有 EXP-006 的 FM8 C1/AA/AD 73帧，可直接截取对应 C1+C2 的56帧并复用原测量；先逐项核对来源 fixture、seed、噪声和动作，协议无差异就**不重跑 FM8**。新 AF 每场景从自己 C1 到 AA/AD C2：`8 C1 sampling +1 clean commit+2×8 C2 sampling=25` forward、3 decode。另以已保存 FM8 C1 为共享历史，由 AF 自己重建 C1 KV，再做 A/D C2：`1 commit+16 sampling=17` forward、2 decode。因此三场景新增上限 **3×(25+17)=126 forward、15 decode**，而不是把已有 FM8 重算一遍。若预检发现真实协议差异，必须说明差异并由 Judge 另行修订预算；不能暗中追加调用。这是独立拟议账本，不纳入当前任务。实际比较动作符号/幅度、完整人物与场景、后半块残影、边界和端到端时间；不能用 loss、finite-map 一致性或某一帧截图代替。结果不用于挑 checkpoint、改样本或继续调参。两验证 episode 太少，不作统计泛化结论。

**门 4 / DMD：** 只有新 student 在多个固定场景的 generated-history 续写中同时保持基本动作和结构，并且已有清楚的长期 generated-history gap，Judge 才考虑独立 DMD 任务。此前 AF4 与 DMD-lite 失败配置继续归档，不在本计划中恢复。
