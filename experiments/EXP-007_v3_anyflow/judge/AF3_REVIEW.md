# EXP-007/v3 AF3 Judge审核

2026-10-11，Judge。**accept：step32 AnyFlow在共同FM8首窗后的AA/AD73帧、8NFE续写有限可行性通过；画质与严格连续性PARTIAL。尚无整体优于普通FM8的证据。** 本轮收口，不扫checkpoint、步数或学习率，不覆盖正式V3 Baseline。

## 已完成与独立检查

AF2累计32次finite-map更新的配套target/QKV作为唯一评估checkpoint。AF3共35forward（32sampling+3clean prefill/commit）、4VAE、0update。首次导入路径错误在模型forward前退出；修复后获Judge明确重试，失败3.560679秒保留在累计账本。总占用473.000176秒=0.131388938GPU小时，peak allocated26.336456GiB，低于0.35GPUh/44GiB预算。所有运行已结束。

[独立审计](af3_completed_audit.json)核对18项冻结来源、配对step32、全5个成功阶段与失败账本、缓存来源SHA和50层indices、输入noise/prompt/sigmas与FM8匹配、C2共同clean历史、C3各自生成历史、历史raw RGB逐值不变、endpoint/缓存/视频SHA、全片MP4解码与24fps递增PTS。全部通过。初次Judge审计缺系统ffprobe，改用已安装PyAV完整解码完成同一检查，没有新增GPU。

Judge检查四块全部68张新增帧、相邻边界以及原分辨率代表帧；见[逐帧判断](af3_visual_notes.json)和本目录AF3_*图像。人物和停车场持续可辨，未出现持续主体消失或全场景崩溃；有边界姿态/视角跳变，AA第三块下肢透明残影持续，AD第三块局部双轮廓且动作幅度较弱。

| 路径 | FM8 C2水平flow | AF8 C2 | FM8 C3 | AF8 C3 |
| --- | ---: | ---: | ---: | ---: |
| AA | +0.8458 | +0.6502 | +0.5052 | +1.0064 |
| AD | −0.3206 | −0.6972 | −0.8265 | −0.1494 |

方向及切换响应可辨；光流仅辅助判断，不是控制准确率，也不是幅度越大越好。AF第三块边界灰度MAD为AA6.06、AD11.28，块内均值约3.99、2.53；与视觉观察共同提示边界问题仍未解决。普通缺陷按可行性尺度接受，不因PARTIAL无限追加训练。

## 比较范围

首39帧与clean C1借用EXP-006新FM8，各模型用自身权重创建raw KV。C2共享历史/动作/噪声/8NFE，C3分别续自己的C2。应称**AF continuation**，不能称从初始首窗全程AF已验收。训练使用原V3 FM30生成的C1/C2，评估C1来自FM8；这一历史分布差别保留。单停车场/seed13、73帧，不是独立泛化、长时滑窗或成熟控制质量。

训练和target-time条件共同变化，不作纯训练单因素归因；未测公平E2E速度或首帧时延。普通FM8已是实用有限基准，本轮AF训练证明新协议student可训练且少步续写基本可用，尚不能证明额外训练的视觉收益。

## 证据与下一步

- [Worker AF2报告](../worker_report_v3_af2.md)、[AF2 Judge审核](AF2_REVIEW.md)
- [AA AF73原片](../artifacts/af3_raw/AA/rollout_73.mp4)、[AD AF73原片](../artifacts/af3_raw/AD/rollout_73.mp4)
- [AA FM8/AF8对比](../artifacts/af3_videos/AA_FM8_vs_AF8_continuation_73.mp4)、[AD FM8/AF8对比](../artifacts/af3_videos/AD_FM8_vs_AF8_continuation_73.mp4)

该student可作为独立DMD工程pilot起点。下一任务EXP-008先检验真实8-map链的完整DMD反向、角色隔离及成本；仅一个cycle，不预设画质收益。正式任务书、代码来源与CPU预检通过后独立放行，后续有限训练和视频仍需Judge根据实测ROI决定。
