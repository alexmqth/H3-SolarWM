# Breakthrough 02：RGB-consistent anchor 修复人物分解

## 问题

旧 Stage2-lite 视频在后续 chunk 出现人物透明、分裂和 ghosting。单纯更换 solver 或增加 anchor 混合不能可靠解释这个现象。

## 根因

训练 visual tail adapter 时使用的是：generated prefix -> VAE decode RGB -> last RGB frame -> H3 image branch encode。旧推理却把 generated latent tail 直接 patchify 成第二 anchor。训练和推理的 conditioning protocol 不一致。

## 方案

固定 causal mask、KV、chunk 和 solver，只把第二 anchor 改成 `dynamic_last_frame_rgb_dual`：生成 prefix 解码为 RGB，再通过 H3 image branch encode，并与固定 initial image anchor 一起输入。使用 tail16 visual QKV adaptation 和原始 H3 chunk-endpoint target。

## 证据

- `evidence_old_vs_rgb_anchor_39.mp4`：旧 latent-only 与 RGB-consistent 39-frame 对比；
- `evidence_W_124.mp4`：RGB anchor 修复后的原始/causal W 长片；
- `VISUAL_DRIFT_REPAIR_REPORT.md`：训练配置、显存、时间、39/124 帧检查和 Stage2-lite 集成结果。

修复后 39-frame A/D 和 124-frame W/A/D 的人物、停车场结构保持到视频末尾。W 长片约 31,570 MiB GPU peak、13.19 GiB CPU raw-KV、约 709 s sampling。

## 结论

视觉稳定性问题主要来自 anchor protocol mismatch，RGB-consistent anchor 有效修复了人物分解。A/D signed flow 仍错误，因此这个突破不能称为 action control 恢复。
