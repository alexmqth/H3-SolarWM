# 5分钟答辩：两条并列路线的取舍与统一目标

**0:00–0:45：目标与因果化代价。** 播放[V0 vs V1](00_comparison_gallery/V1_vs_original.mp4)。我们要把SolarWM的causal/KV/少步思路迁到H3-World。V1工程可执行，但A/D近乎停滞；工程实现与能力验收分开。

**0:45–1:35：为什么action信息流重要。** Original是Single-Egress，多层video传播负责间接传动作；causal action prefix增加past-action直读，来自fresh prefix而非缓存action。CPU cache保存raw video K/V，clean commit和滑窗。受控数值P0等价不表示动作好，无persistent KV的strict图已经失配。

**1:35–2:30：同一问题的两条修复路线。** 播放[V2a vs V2b](00_comparison_gallery/V2a_vs_V2b.mp4)。V2a修复RGB图像条件并训练visual adapter，支持strict causal/KV，124f结构改善，A/D失败；V2b从Original恢复Single I0/native time，用Same-σ和局部双向重算，无新增训练也无persistent KV。两者不是升级关系，步数/history/cache也不匹配，比较的是能力取舍。

**2:30–3:25：正结果与失败边界。** 播放[V2b四路径](00_comparison_gallery/V2b_four_paths_56.mp4)，RGB39进入第二块：自身history，无GT reset，AA/AD/DA/DD局部方向和人物结构成立。随后展示[V2a 20秒失败](V2a_rgb_anchor/videos/V2a_long20s_failure.mp4)后段；V2a只证明124f视觉相对稳定，V2b只证明56f局部，不能说两者都解决长期崩坏。

**3:25–4:15：少步与训练探索没有神奇修复。** 8/chunk×8=64 noisy+8commits，对照Original30次长序列forward，不能算30/8加速。RGB decode/re-encode、CPU offload和重算成本不同。ordinary FM、真实ABot、AnyFlow16/64/128/136和DMD-lite都保留负结果；不是完整Stage1/2成功。

**4:15–5:00：未来V3的技术判断。** V3要统一高效严格因果生成、视觉稳定和动作保真，但不能直接拼接两个checkpoint。需要一个不依赖history/current双向重算仍有正确动作能力的causal backbone，再做AnyFlow和on-policy DMD。目前没有V3模型/视频，也没有统一硬件重复均值的效率结论。

[导航](README.md) · [维度对照与路线](roadmap.md)
