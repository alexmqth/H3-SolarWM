# EXP-011/v1 G1 Worker 报告：两个场景的原生首窗

2026-10-11 06:17 HKT。四个预定配置均已完成，且 Judge 已逐帧审核并分别放行 G2；这是首窗有限可行性证据，不是动作切换或续写验收。授权及独立审计见 [G1 review](judge/G1_REVIEW_AND_G2_RELEASE.md)、[final audit](judge/G1_FINAL_REVIEW.json) 和 [冻结账本](judge/G1_budget_frozen.json)。

## 固定输入和实际执行

两张已固定的 ABot validation PNG，来源及静态文字见 [source manifest](source_manifest.json)。P1 已从原 PNG 生成 native full37 / Single I0 fixture；工业、村落 fixture SHA-256 分别为 `9f404c910c31e68b319cf6029d7d59fa48229d9db14df395a2890cc32057107d`、`eb7590213fb7f3971f3e6d345bce185ab8f9b6fba757f43a2123b65631478775`。本阶段未读取旧 RGB dual-anchor encoded 输入，没有 GT history、teacher endpoint 或新增训练。

实际环境为 GPU0、冻结 H3 + released action LoRA、`h3_fp32`、native sigma/shift 2.22、Global RoPE、strict chunk-causal、current-prefix feedback、seed13。同一场景的 FM30/FM8 使用相同原始初噪声、I0 和静态文字，各自独立生成首 12 latent、解码 39 RGB；首窗没有 KV commit。运行命令均为 `run_exp011.py --stage G1 --scene <scene> --method <method> --gpu 0`，完整环境、原日志和逐配置结果保存在 [artifacts/G1](artifacts/G1/)；原始 endpoint 和 RGB numpy 留在 `H3-World/outputs/EXP-011_v3_scene_transfer/G1/`。

| 场景/方法 | Sampling forward | Decode | GPU wall s | Peak allocated GiB | 全帧视觉复核 |
|---|---:|---:|---:|---:|---|
| industrial/FM30 | 30 | 1 | 194.440 | 25.086 | 人物、植被、桥与工业建筑全程可辨；细节偏软 |
| industrial/FM8 | 8 | 1 | 74.972 | 25.086 | 人物和场景可辨；人物与植被略软 |
| village/FM30 | 30 | 1 | 219.922 | 25.091 | 中段前景树遮挡人物，之后重现；局部细节偏软 |
| village/FM8 | 8 | 1 | 75.220 | 25.091 | 人物重现，但树叶交叠残影与地面颗粒感更重 |

推理总账本：**76 sampling forward、0 commit、4 decode、577.338 GPU 秒**；本阶段最高 allocated 25.091 GiB。四条视频均为 832×480、24 FPS、39 帧；PyAV 完整解码与唯一 PTS 检查通过。Judge 的独立 CPU 审计核对了 fixture、原始噪声、sigma、坐标、结果哈希与帧数，四配置均 PASS。两条同场景横向视频在 [comparisons](artifacts/G1/comparisons/)；G1 没有生成历史，同场景 FM30/FM8 使用匹配的 fixture、I0、初噪声、权重和协议，只改变 native sampling steps。后续 G2 则分别接续自身生成的 C1，属于闭环方案对比，不是同一历史状态下的纯步数消融。

四配置均为 **first-window PASS_LIMITED / quality PARTIAL**。单条 A 首窗不能确定 A/D 方向控制，persistent KV 也还没有在 G1 使用。G2 已分别获批，下一步从各自 C1 clean KV 分叉 AA/AD 并检查 17 张新增帧。若出现协议、数值、资源错误或持续结构崩溃，按任务书停止对应后续配置，不自动重试。
