# EXP-011/v1 G2 Worker 报告：两张 ABot 初图的自身历史 A/D 续写

2026-10-11 06:42 HKT。四个已分别放行的 G2 配置全部执行完成，Worker CPU 协议审计全部 PASS；Judge 已逐配置检查新增画面并在 [最终验收](judge/FINAL_REVIEW.md)中接受**两块、56 帧的有限可行性**，画质均为 **PARTIAL**。本报告保留 Worker 的实际执行与证据；正式主线结论和 GitHub 发布由 Judge 维护。

## 研究问题与固定协议

检验已在停车场建立的 V3 严格 chunk-causal + persistent raw KV 协议，在两张预先固定的 ABot validation 初图上，能否从**各自生成**的39帧首窗继续17帧，并在同一个 C1 状态下对当前动作 A/D 作出区分。工业场景来源 `A_1140.png`，村落来源 `D_1750.png`；两图都是游戏录屏的 validation episode，不称真实世界摄影，也不声称 H3 预训练绝对未见。来源、split、静态文本和哈希见 [source manifest](source_manifest.json) 与 [P0 报告](P0_REPORT.md)。

原始 H3 + released action LoRA，无新增训练；native full37 packed horizon、Single I0、own-action routing、current non-action prefix feedback、Global RoPE、strict chunk-causal、`h3_fp32`、native flow shift 2.22、seed13、832×480/24 FPS。C1 为12 latent/39 RGB；每种方法单独从相同初噪声生成 C1，再 sigma0 clean-commit 自身 C1 的50层 CPU raw KV；同一缓存和 C2 初噪声分叉 AA/AD，C2 为5 latent/17 RGB，到56帧。FM30 与 FM8 是普通 FM 的30/8步采样，**FM8 不是 AnyFlow**；两者 G2 使用各自 C1，因此不是同 KV 状态的纯步数消融。原始双向 H3 未在这两张初图上生成匹配视频，不能据此量化相对 Original 的动作保真率。

## 实际调用与资源

四配置依次在实时空闲的 GPU0 执行，均使用冻结 `run_exp011.py` 和对应 `judge/G2_*_APPROVED.json`，未触发自动重试、调参或额外 GPU 任务。G1 和 G2 的推理总账本为 **228 sampling forward + 4 clean commit = 232 forward、12 VAE decode、1781.710 GPU 秒**，低于 4500 秒上限；P1 输入编码另为6 text/2 image encode、70.565 GPU 秒。全部无 backward/update。G2 每配置缓存均为 **6,799,104,000 bytes（约6.33 GiB）**，峰值 GPU allocated 26.119 GiB；P1 encoder 阶段的峰值另为40.576 GiB。完整逐调用 [最终账本](artifacts/G2/budget_final.json) 与各配置原日志留存。

| 场景/方法 | C1 G1 wall s | G2 wall s | G2 forward / decode | Peak G2 GiB | C2 AA/AD 光流辅助值 | 画面复核 |
|---|---:|---:|---:|---:|---:|---|
| industrial / FM30 | 194.440 | 463.744 | 61 / 2 | 26.119 | +26.70 / −43.98 | 人物与工业场景保留；细节偏软 |
| industrial / FM8 | 74.972 | 184.029 | 17 / 2 | 26.119 | +28.20 / −38.27 | A/D 分叉可辨；软化略重 |
| village / FM30 | 219.922 | 365.999 | 61 / 2 | 25.909 | +68.09 / −94.49 | 人物、房屋和石路保留；树木短暂遮挡 |
| village / FM8 | 75.220 | 171.107 | 17 / 2 | 25.909 | +87.82 / −44.23 | A/D 分叉可辨；AA 树叶/人体边缘重影更明显 |

光流值来自 [CPU 审计脚本](audit_g2_cpu.py) 对 C2 16 对相邻帧的中央区域 Farneback 水平位移中位数求和，仅辅助展示方向分叉；它不是对角色动作正确性的独立标签，也不是与 Original H3 的等价评分。边界 MAD 与段内 MAD 的原值在逐配置 [审计 JSON](artifacts/G2/audits/) 中。上表 wall 包含分阶段模型加载与解码，不是一个常驻服务的完整端到端延迟；G1/G2 也在各自进程里加载模型。30→8 NFE 的计算减少已被测到，但不能据此宣称产品级整体加速比。

## 协议与视频验收

四组均核对了：G1 endpoint/RGB SHA 与逐配置 marker 一致；AA/AD 的 C2 初噪声、sigma、位置完全一致；动作差异仅在当前 C2 action spans；真实50层 C1 cache 的 index0 与字节数；推理时缓存签名不变；两分支发布视频的前39 RGB 与 G1 **逐值相同**；新17帧不同且均有限；全部8条56帧和8条新增17帧 MP4 为832×480、24 FPS、PTS 唯一且可完整解码。Judge 的独立 G2 审计文件在 `judge/G2_*_audit.json`；Worker 的 [四份审计结果](artifacts/G2/audits/) 保留帧差与辅助光流。

全帧视觉复核：工业场景两种步数都能在 C2 保持人物、桥、树木和建筑；AA/AD 产生方向不同的场景/人物运动，边界有姿态与视角调整。村落场景的树木遮挡会影响人物可见性，但人随后重现，未见持续全噪声或场景消失；FM8 的暗部颗粒、树叶和人体交叠残影更明显。四配置均只覆盖**两个 chunk、一个 seed、两个预定场景**，不能推到124帧、更多场景或长期稳定性。

## 视频入口与证据位置

同方法 AA/AD 并排片最能直接看动作干预：

| 场景 | 30-step | 8-step |
|---|---|---|
| 工业 | [FM30 AA vs AD](artifacts/G2/comparisons/industrial_FM30_AA_vs_AD_56.mp4) | [FM8 AA vs AD](artifacts/G2/comparisons/industrial_FM8_AA_vs_AD_56.mp4) |
| 村落 | [FM30 AA vs AD](artifacts/G2/comparisons/village_FM30_AA_vs_AD_56.mp4) | [FM8 AA vs AD](artifacts/G2/comparisons/village_FM8_AA_vs_AD_56.mp4) |

同场景、同动作序列的 FM30/FM8 横向片：[工业 AA](artifacts/G2/comparisons/industrial_AA_FM30_vs_FM8_56.mp4)、[工业 AD](artifacts/G2/comparisons/industrial_AD_FM30_vs_FM8_56.mp4)、[村落 AA](artifacts/G2/comparisons/village_AA_FM30_vs_FM8_56.mp4)、[村落 AD](artifacts/G2/comparisons/village_AD_FM30_vs_FM8_56.mp4)。各配置的原始56帧、17帧视频、全帧图、运行日志与结果 JSON 在 [G2 artifacts](artifacts/G2/)；大 endpoint、6.4 GB/配置的 cache 和 raw RGB tensor 保留在本机 `H3-World/outputs/EXP-011_v3_scene_transfer/G2/`，不入 Git。文件 SHA 与对应原位置见 [artifact manifest](artifacts/G2/manifest.json)。

## 结论与边界

本轮支持：在两张固定的新初图上，V3 causal FM30 和普通 FM8 都能完成两块自身历史续写；persistent KV 实际复用，AA/AD 同状态干预引出可辨的反向运动，人物与场景短程保持基本可用。它并未证明成熟画质、跨场景统计泛化、长期无漂移，也未完成 AnyFlow 或 DMD。村落 FM8 的重影说明少步方案的视觉代价仍需单独处理；不在本任务中追加训练或 C3。
