# 长视频与动作/视觉取舍：工程验证及局限

**视觉结论：20 秒 causal 稳定性失败。** 约 10 秒开始严重雾化，15 秒后人物和场景难以辨认。本记录属于长度测试与限制定位，不能作为长视频画质突破展示。

问题：仅保存视觉稳定版会掩盖 action control 退化；仅看 5 秒也无法判断长时漂移。

处理：同时保留旧 fixed-mix action checkpoint 与 RGB visual checkpoint，制作原始 H3／旧 causal／新 causal 的 A/D 三列对照；在同一 RGB visual checkpoint 上真实生成 243 和 481 帧，并生成同长度原始 H3 30-step 对照，逐长度验证首帧、prompt、video/audio noise 指纹一致。

- [三列 A/D 对照](../../meeting/action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)：旧版方向符号较好但视觉漂移，新版结构较完整但 A 的方向错误。
- [20 秒原始/causal 并排](../../meeting/long_horizon/original_vs_rgb_visual_W_20s_481f.mp4)：完整 20.042 秒原始/causal 对照，不循环、不截掉后段。

解决的工程问题：同一 causal checkpoint 完成 29 chunks、232 noisy forwards、29 clean commits，5-chunk CPU KV 峰值保持 13.19 GiB；原始 H3 长视频的 eager mask 临时张量 OOM 用等价的编译构造解决。

未解决的问题：generated-history 漂移、纹理/人体细节退化和 A/D action geometry。这是长度扩展和对照证据，不是新的训练突破，也不是长视频质量 PASS。

完整数字和视觉观察见 [长视频报告](../../meeting/long_horizon/README.md)，两套 checkpoint 的配置见 [动作/视觉对照](../../meeting/action_vs_stability/README.md)。
