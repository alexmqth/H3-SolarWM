# EXP-012 工业阶段审核与村落放行

2026-10-11 07:01 HKT。工业GPU运行结束，17forward/2decode、累计168.848117 GPU秒，allocated峰26.223347GiB；0训练/额外编码。

独立真实cache和视频审计PASS：AF自己在共同FM8 C1上建立50层index0 raw KV，6,799,104,000 bytes；对比FM8真实cache，全50层K/V内容不同而RoPE逐值一致。同C1/noise/action/sigma/position与FM8匹配，旧39RGB不变，完整56/17帧解码通过。只说明实现与来源正确。

Judge看完全部34新帧与原分辨率FM8/AF8四格末帧。人物/桥/建筑保留，画质PARTIAL；AA运动可用。AD虽与AA姿态/轨迹不同，17帧内反向响应不清楚；中央ROI中值flow累计AF8 AA+28.14/AD+11.07，对照FM8+28.20/−38.27，仅辅助但与弱切换观察一致。**动作切换PARTIAL，未证明工业场景联合收益；不把这组写成完整动作通过。** 没有持续结构崩坏。

保留另一预先固定场景的有限机会，批准GPU0村落一次1commit+16sampling/2decode，总任务预算不重置（余1091.151883秒），09:00截止，0训练/新编码，无C3/重试/调参。marker见G1_village_APPROVED.json；完成后本任务收口，若仍无收益优先普通FM8。
