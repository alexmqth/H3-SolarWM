# Breakthrough 05：修正 endpoint state mismatch 后仍未恢复动作

## 问题

第一版 endpoint target 把独立 D trajectory 的 endpoint 用在 A-generated history 上，存在 state mismatch。这个协议不能判断 endpoint supervision 是否有效。

## 修正

对 A rollout 使用 A own-history endpoint target，对 D rollout 使用 D own-history endpoint target；保持 39 frames、3 chunks、RGB dual anchor、generated history、8 steps/chunk、CPU raw KV、seed=13 不变，只改变 endpoint target 的 state 配对。

## 结果

修正后的 own-history endpoint + paired QKV 结果：

    flow(A) = -0.8086
    flow(D) = -0.6893
    A-D     = -0.1194
    latent delta cosine = 0.0655

人物和停车场结构保持，但 `flow(A)>0, flow(D)<0, A-D>1.0` 仍全部未满足。对应视频为 [own-history 端点视频](../../meeting/action_geometry/stage2_action_qkv_own_endpoint_AD_39.mp4)，报告为 [ACTION_QKV_OWN_ENDPOINT_REPORT.md](../../reports/endpoint_target/ACTION_QKV_OWN_ENDPOINT_REPORT.md)。

## 结论

state mismatch 已被排除，但单一 generated state 的 endpoint target 仍不能把 causal score field 旋回原始 H3 的 image-space action geometry。当前最终边界是：causal/KV 和视觉稳定性已完成，动作恢复需要多状态、多 seed 或更强的 causal action topology/distribution training。
