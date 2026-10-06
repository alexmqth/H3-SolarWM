# Breakthrough 01：causal chunk rollout 和 persistent KV 跑通

## 问题

原始 H3-World 可以生成 action-conditioned 视频，但每次推理都按完整 horizon 计算，不能直接复用已生成历史。需要确认 SolarWM 风格的 chunk-wise causal attention 和 KV cache 能否接到 H3-World 的 directed action routing 上。

## 方案

- 每 chunk 使用 5 个 latent frames；
- history window 设为 5 chunks；
- 每个 chunk 使用 8 solver steps；
- denoise 后用 clean latent 做一次 clean KV commit；
- 每层保存 raw K/V，历史 cache 放在 CPU；
- action rows 使用 causal prefix，并保留 action feedback edge；
- generated history 作为后续 chunk 的输入。

## 证据

`evidence_W_original_vs_causal.mp4` 是原始 30-step 与 causal W 的并排对比；`evidence_action_grid_124.mp4` 是 124-frame W/S/A/D 总览。124-frame rollout 完成 8 chunks、64 noisy forwards、8 clean commits，replay error 为 0，视频可以完整解码。

## 结论

H3-World causalization 在工程上可行，KV cache 不是只添加了一个静态 mask，而是真正参与历史复用。这个突破只证明 causal execution，不证明少步质量或 action geometry 已经保留。
