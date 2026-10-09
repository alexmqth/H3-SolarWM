# 首条真实验证条件的 Original30 参照

片段 `118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140` 来自本轮 held-out episode；保持保存的同首帧、静态prompt、真实逐帧联合动作、seed13及视频/音频噪声。

原始 H3 30-step 已完整生成39帧，H264/yuv420p/24fps全帧解码通过。全部39帧contact sheet和0/8/16/24/30/38对应GT帧已静态检查：人物到最后仍存在，树林/建筑/道路结构连续，没有此前人物分解或严重雾化。生成轨迹和背景细节并不逐像素复制GT。这是静态帧检查，不声称已人工实时播放。

- 30次整段denoiser前向，采样207.00s，包含模型加载与输出编码的墙钟231.64s。
- GPU allocated peak25928.91MiB；没有持久化CPU KV。
- Farneback平均水平flow +2.2888；这条真实动作包含forward、strafe left、camera pan left，**不能作为纯A控制正确率**。
- 第17/34帧RGB帧差20.2753/17.2324；对Original这些仅是与causal边界对应的时间点，不是模型内部chunk边界。移动户外画面不能套用停车场样本的绝对MAD阈值。
- 单次共享主机、预先缓存conditioning，不作warmup均值或speedup结论。

原始输出位于 `eval/original/generated_30/118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140/`。
后续还需要另一条Original参考、step00因果基线及48更新后的GT/generated-history对照。此结果不代表新的causal checkpoint通过验收。
