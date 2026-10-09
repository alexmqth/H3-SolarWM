# 视频比较范围与公平性

所有新视频只从已生成MP4取真实RGB帧：24fps、不补帧、不慢放、不循环；等比例缩放留边、横向左baseline/另一方案、右当前方案（V2a vs V2b是并列能力比较）、统一0-based RGB帧号。H.264/YUV420P/faststart，逐帧解码验证。静音是原生成协议；没有合成画面。

| 比较 | 显示帧段 | 相同项 | 不同项 / 限制 |
|---|---|---|---|
| Original vs V1 | [0,124) | 停车场、首图/prompt记录、常量A/D、seed13、832×480源 | 30整段 vs 8/chunk，Single I0 vs latent dual，prefix时间/feedback/attention，不能解释为仅加KV |
| Original vs V2a | [0,124) | 同场景/首图/prompt/action/seed/源分辨率；旧收据有噪声一致性说明 | visual+action adapters、RGB dual、routing/时间/采样协议；Original非GT |
| V1 vs V2a | [0,124) | chunk5、history5、8/chunk、CPU KV、常量动作、seed13 | 无新增adapter vs visual+action训练，latent vs RGB，own/off vs causal/on；联合方案比较 |
| Original/V1/V2a vs V2b | [0,56) | 常量A或D、同停车场首图/seed13、源832×480、24fps | full124生成后裁56 vs12+5可见窗口；精度、时间、anchor、routing、history、步数不同；V2b不继承V2a权重 |
| V2b四路径 | [0,56)；39处边界 | 每行同自身generated first12，当前A/D反事实，共享保存噪声 | 只有第二块；跨两行history本来不同，不直接相减作为同状态velocity |

Same seed不自动证明noise逐字节一致，尤其不同张量长度。V2b来自固定full37 fixture，源protocol保存输入与runtime hash；V0/V2a旧比较有自己的来源核验。本次只核验文件与收据，没有加载大tensor伪造新的张量匹配证明。

**MISSING_MATCHED_VIDEO**：没有找到与V2b A→D / D→A相同39帧切换点的Original和V2a完整56f视频。不会使用恒定A/D冒充这些切换基线。V2b与Original/V2a仅比较AA/DD；其余两条单独展示。

V2b的首39帧是已保存的自身生成首段，随后追加RGB39:56；不是GT reset，也不是把Original full-horizon输出拷作history。VAE重新decode可能改写过去末5帧，本展示按原实验冻结已发帧，保留边界跳变。

V2a与V2b是同一问题的并列修复路线，无继承关系；V2b跨版展示是能力比较，不能归因Same-σ单项；V2b不是strict chunk-causal attention或persistent KV的成功证据。所有完整源片同时保留，裁剪对比不会隐藏V2a的124/481f负结果。

视频中的时间来自原运行收据：V0/V1/V2a标完整124f E2E，即使展示只裁56f；V2b标第二块sampling并注明重用首窗。均不是制作MP4所花时间。

[画廊](00_comparison_gallery/README.md) · [指标口径](METRICS.md) · [首页](README.md)
