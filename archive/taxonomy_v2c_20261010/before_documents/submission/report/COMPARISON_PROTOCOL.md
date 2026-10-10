# 视频比较范围与公平性

所有新视频只从已生成MP4取真实RGB帧：24fps、不补帧、不慢放、不循环；等比例缩放留边、横向左baseline/另一方案、右当前方案（V2a vs V2b是并列能力比较）、统一0-based RGB帧号。H.264/YUV420P，逐帧解码验证。静音是原生成协议；没有合成画面。

| 比较 | 显示帧段 | 相同项 | 不同项 / 限制 |
|---|---|---|---|
| Original vs V1 | [0,124) | 停车场、首图/prompt记录、常量A/D、seed13、832×480源 | 30整段 vs 8/chunk，Single I0 vs latent dual，prefix时间/feedback/attention，不能解释为仅加KV |
| Original vs V2a | [0,124) | 同场景/首图/prompt/action/seed/源分辨率；旧收据有噪声一致性说明 | visual+action adapters、RGB dual、routing/时间/采样协议；Original非GT |
| V1 vs V2a | [0,124) | chunk5、history5、8/chunk、CPU KV、常量动作、seed13 | 无新增adapter vs visual+action训练，latent vs RGB，own/off vs causal/on；联合方案比较 |
| Original/V1/V2a vs V2b | [0,56) | 常量A或D、同停车场首图/seed13、源832×480、24fps | full124生成后裁56 vs12+5可见窗口；精度、时间、anchor、routing、history、步数不同；V2b不继承V2a权重 |
| EXP-001 Original vs V2b持续A/D | [0,124) | 首图、seed13、full37视频/音频noise、prompt/image position/action输入均核对 | Original联合音视频整段30步；V2b固定audio条件、六块各30步、Same-σ全历史重算；无KV，无公平speedup |
| V2b四路径 | [0,56)；39处边界 | 每行同自身generated first12，当前A/D反事实，共享保存噪声 | 只有第二块；跨两行history本来不同，不直接相减作为同状态velocity |

Same seed不自动证明noise逐字节一致，尤其不同张量长度。V2b来自固定full37 fixture，源protocol保存输入与runtime hash；V0/V2a旧比较有自己的来源核验。旧56帧素材整理仅核验文件与收据；EXP-001另有Original输入张量匹配审计，见实验original_source_audit.json。

**MISSING_MATCHED_VIDEO**：没有找到与V2b A→D / D→A相同39帧切换点的Original和V2a完整56f视频。不会使用恒定A/D冒充这些切换基线。V2b与Original/V2a仅比较AA/DD；其余两条单独展示。

V2b的首39帧是已保存的自身生成首段，随后追加RGB39:56；不是GT reset，也不是把Original full-horizon输出拷作history。VAE重新decode可能改写过去末5帧，本展示按原实验冻结已发帧，保留边界跳变。

V2a与V2b是同一问题的并列修复路线，无继承关系；V2b跨版展示是能力比较，不能归因Same-σ单项；V2b不是strict chunk-causal attention或persistent KV的成功证据。所有完整源片同时保留，裁剪对比不会隐藏V2a的124/481f负结果。

视频中的时间来自原运行收据：V0/V1/V2a标完整124f E2E，即使展示只裁56f；旧56帧V2b片标第二块sampling并注明重用首窗；新124帧片标73→124增量wall。均不是制作MP4所花时间。

[画廊](00_comparison_gallery/README.md) · [指标口径](METRICS.md) · [首页](README.md)

## EXP-001新增124帧范围

EXP-001 / v3已验收：单停车场、seed13，持续A/D六窗口124帧（5.17秒）具备可辨响应与基本人物/场景结构。保留AA RGB72→73姿态跳变、动作节奏不均和局部细节软化；DD后段靠近画面下边缘。AD/DA仅到73帧，DA第三块flow轻微反号；切换和泛化未通过。无persistent KV、无公平加速结论，V3后续以EXP-002/003完成有限可行性验收。 新片左侧为2026-10-01-21匹配Original A/D参考，右侧为EXP-001最终124f。Original计时为完整E2E 478.4/454.6s，V2b为复用前73帧后三块增量1216.8/1194.9s。两种口径不能相除作speedup。A/D总览为两段依次播放，共248帧，不是单条248帧rollout。


## EXP-003 / V3正式对比的边界

Original/V3 AA、V2b/V3 AA均完整124帧/24fps；V2b/V3 AD仅共有73帧，Original A→D匹配缺失。原片保留AA瞬态严重形变。V3与V2b同时改变拓扑、history sigma和缓存，不是单因素消融；后续生成历史不同。

V3 AA增量73→124占卡798.95秒，包含加载、cache IO、sampling、解码、编码与人工检查等待；Original整片E2E478.4秒、V2b对应增量1216.8秒，口径不同不能求统一speedup。每块30步sampling的观测对照见[V3页面](V3_efficient_causal/README.md)。
