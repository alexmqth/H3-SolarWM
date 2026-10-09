# 停车场 FM48 30/8步完整人工评审

2026-10-08 22:35。静态检查两动作所有0–38帧contact sheets及Original/FM0/FM48的0/8/16/24/30/38对应帧；没有实时播放评审。

| 配置 | flow(A) | flow(D) | A−D | 视觉观察 |
|---|---:|---:|---:|---|
| Original30整段 | +1.181253 | −0.842124 | 2.023377 | 人物连续、左右运动不同 |
| causal0 30/chunk | −0.195516 | −0.186950 | −0.008565 | 人物/场景大体保留，A/D近同向 |
| FM48 30/chunk | −0.182680 | −0.163995 | −0.018685 | 没有明确动作恢复，人物大体完整 |
| causal0 8/chunk | −0.021397 | −0.031908 | 0.010510 | 约20–22帧后重影 |
| FM48 8/chunk | −0.025848 | −0.043304 | 0.017456 | 约22帧后人物透明叠影、车库轮廓叠影仍在 |

8步的MAD/边界帧差下降不能当作质量提升：FM48 A灰度MAD3.0808→2.6163、boundary5.0099→3.3061；D灰度MAD3.0158→2.6526、boundary5.3184→3.7995。后段仍虚化，光流几乎无方向区分。30步提高画面完整性但不恢复A/D；不是AnyFlow或Stage2 checkpoint。

causal0/48使用相同h3_fp32、RGB dual、seed13、相同输入noise/action/初帧，每条39帧5+5+2 chunks，CPU raw KV6484.13MiB。30步90 noisy+3 commits；8步24 noisy+3 commits。Original旧legacy精度且条件协议不同，与Original只能作pipeline对照。单次共享主机时延不可宣称speedup。

[30步完整视频](report/parking_step48_30step/original_causal_AD.mp4) · [8步完整视频](report/parking_step48_8step/original_causal_AD.mp4) · [39帧/编码/faststart验证](trained_partial_archive_video_validation.json)

结论：本次有限预算真实FM48没有通过停车场action/8-step视觉gate。它不证明普通FM永远不可行；不延长原预算。
