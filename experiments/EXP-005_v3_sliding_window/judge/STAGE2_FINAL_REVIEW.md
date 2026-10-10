# Judge阶段二验收：EXP-005 / v2

2026-10-10 HKT。**任务完成，SW-G通过158帧有限可行性验收；SW-L工程正确、生成质量PARTIAL，无已证实的联合收益，归档当前无训练Local方向。** 正式V3 Original Feasibility Baseline继续冻结为124帧参考。没有新增训练，没有C9，没有额外scene/seed或调参。

## 1. 已执行与独立审核

用户明确批准开始实验后，Judge按G0→G1→L1逐段发出绑定runner/config/source manifest的授权。Worker提交实际结果后，Judge独立核对GPU预算、源码/输入身份、缓存索引与容量、最终latent文件摘要、所有新增RGB帧及边界帧、历史RGB逐值不变、MP4帧数/24fps/PTS。全部八段新增17帧已逐帧查看；原分辨率末帧和带标签并排视频保留。

| 项目 | SW-G | SW-L |
| --- | --- | --- |
| 真实淘汰/精确5祖先/50层 | PASS | PASS |
| 采样不改历史cache，旧RGB不变 | PASS | PASS |
| A/D方向与首次切换响应 | 测试范围内PASS | 方向信号保留，测试范围内PASS |
| 主体与场景 | 主体/停车场基本可用 | 主体可辨，场景稳定性更差 |
| 画质/连续性 | PARTIAL：边界跳变，D-C8持续半透明拖影 | PARTIAL：重复几何、透视/亮度跳变、彩色/透明残影 |
| 研究决策 | 保留为后续滑窗参考候选 | 当前无训练Local路线归档，不替代G |

这个决策采用可行性尺度，不以成熟画质要求否定Global。Local的有限负结果说明本次无训练切换没有收益，不证明经过专门训练的Local协议永远无效。

## 2. 首次淘汰前严格回归

G0在两个真实33B状态（sigma1与0.5）比较旧入口与SW-G：maxabs、relative RMS均0。完整30-step C6重放endpoint与冻结EXP-003逐值相等，全部124RGB逐像素一致，包含旧107帧及新增17帧。精度为冻结h3_fp32边界/Transformer bf16，PyTorch2.10.0+cu128、CUDA12.8、L40、默认SDPA调度；没有发现需要放宽阈值的差异。

[G0独立验收](../stage2/g0_judge_review.json)记录完整证据。原EXP-003结束时末C6没有commit；本轮G1才首次提交C6并淘汰C1。

## 3. 淘汰后的动作与画面

全部路径共享冻结AA124历史、I0和新增噪声。C7比较是同历史下的A继续/D切换；C8各自读取自己的C7，是闭环对照。

| 位置方案 | A C7 flow | D C7 flow | A C8 flow | D C8 flow |
| --- | ---: | ---: | ---: | ---: |
| Global | +0.8193 | −1.5745 | +0.7676 | −1.1853 |
| Local | +1.1204 | −0.5977 | +0.3429 | −1.2691 |

单位为水平光流px/frame，仅作动作辅助。两种位置都保留正A/负D信号。Local不是整体动作失效，但C7分离度没有改善，完整画面出现更多重复场景几何、闪变及块内跳变；不能只凭较小的某个boundary MAD宣布提升。Local块内gray MAD约9.0–9.7，Global约4.6–4.9，作为视觉观察的辅助记录而非质量分数。

固定**同一G1-A C7缓存、同一C8 noisy state、sigma0.5、同一动作**的Global/Local诊断得到relative RMS差异**0.182514**、maxabs**2.003342**，历史cache不变。Local确实改变计算，并非无损平移；这不等于更优生成能力。

[全部逐帧视觉笔记](visual_notes.json) · [G1独立账本审计](G1_audit.json) · [L1独立账本审计](L1_audit.json) · [三个协议与RoPE/prefix/camera解释](STAGE2_PROTOCOL.md)

## 4. 缓存、成本与实际预算

所有50层C7使用C2–C6，C8使用C3–C7；历史video KV固定**14,164,800,000 bytes（约13.19 GiB）**。只读采样不变，commit后准确淘汰。旧32-latent历史约18.131 GB，淘汰首12-latent块后变为25-latent历史。

| 阶段 | Sampling | Commit | Diagnostic | 总forward | VAE | 占用秒 | GPU小时 | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 30 | 0 | 4 | 34 | 1 | 309.111 | 0.085864 | 25.598 |
| G1 | 120 | 3 | 0 | 123 | 4 | 1068.666 | 0.296852 | 26.121 |
| L1 | 120 | 2 | 2 | 124 | 4 | 1007.969 | 0.279991 | 26.621 |
| 合计 | 270 | 5 | 6 | **281** | **9** | **2385.746** | **0.662707** | **26.621** |

低于1.70GPU小时上限；从G0账本开始至L1结束elapsed约48.49分钟，低于3小时。全部单GPU0顺序执行，其他占卡进程未触碰；0训练、0重试。CPU进程RSS峰值G1约91,512 MiB、L1约104,275 MiB，不能把13.19GiB历史KV当成整体CPU内存。所有逐块sampling/commit/decode/总时长、GPU allocated/reserved、CPU RSS均见两个独立审计JSON。

阶段计时含加载/保存，sampling包含CPU KV传输，没有单独测传输。G/L阶段有不同diagnostic/commit/I/O，不能据其总耗时宣称Local加速。前124帧为复用历史，本轮成本也不是158帧从零生成E2E。

## 5. 交付视频

- [SW-G：继续A / 切D，158帧](../artifacts/stage2/G1/G1_A_vs_D_158.mp4)
- [Global / Local：A，158帧](../artifacts/stage2/L1/G1_vs_L1_A_158.mp4)
- [Global / Local：D，158帧](../artifacts/stage2/L1/G1_vs_L1_D_158.mp4)
- [全部原始视频/日志/指标归档清单](../artifact_manifest_stage2.json)

对比左Global、右Local；同124帧前缀，C7同历史/噪声，C8各自历史。原片、全新增帧contact sheet与Judge边界检查均保留；大KV/latent/npy只记录外部路径与SHA，不提交权重。

## 6. 研究边界与下一步

验收范围为单停车场seed13，最多158帧/24fps约6.58秒，只有两次真实淘汰。没有验证无限长稳定，也没有跨场景泛化。只有历史video KV有界；prefix、完整输出、VAE解码仍可能增长。显式长fixture保持旧37条件，但不等于默认长packed重建。Local不等价重算历史，且本路径没有SolarWM camera PRoPE。

按ROI停止当前Local方向，保留Global滑窗结果作为后续长窗参考。下一优先提案是**V3-FM8从首窗开始的普通FM减步**，先解决EXP-004仍借用30步首窗的问题；不是继续为Local调参，也不是直接上AnyFlow训练。具体独立预算与V3-AF target-time student、finite-map目标、student自身KV及匹配NFE对照见[FUTURE_ANYFLOW.md](../FUTURE_ANYFLOW.md)。旧AnyFlow checkpoint不作V3-AF完成证据。后续GPU需要新的任务授权，本轮剩余额度不转作训练。

原始runtime结果保留`complete_pending_judge_review`以保持SHA冻结；正式状态以本文件及各stage Judge review为准。
