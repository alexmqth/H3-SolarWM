# EXP-016/v1 — Judge最终验收

2026-10-11 08:48 HKT。**ACCEPT：四段真实视频的native VAE编码、39/56前缀一致性与重建可行性通过。** 这只完成真实视频latent准备，不构成FM/AnyFlow生成能力或完整训练fixture验收。

## 真实执行与独立核查

GPU0一次执行，source video VAE单模型；4image encode、8video encode、4decode、1model load，0text/DiT/audioVAE/训练。实际77.721814826GPU秒=0.021589393GPU小时，allocated峰7.503959GiB，CPU4核affinity，预算内，进程已正常退出。

Judge独立加载全部12个保存tensor并核对来源/hash/shape/finite：I0为[1,24,1,30,52]，full56为17latent，prefix39为12latent。四图full17前12与同一新56RGB之前39编码**全部逐值相等**，max_abs/mean_abs/relative_RMS均0。输入为native float32[0,1]，VAE计算bf16，规范化输出实际float32，未为表面统一修改输出dtype。真实encoder记录33个causal卷积、25个temporal-isolated norm，clip17/token_drop3/tile256/overlap64。[独立实物审核](INDEPENDENT_RESULT_AUDIT.json)。

此证据验证四个固定视频、当前dtype/实现和39/56边界；不是所有VAE或任意长度的形式化因果证明。I0经process_image=True独立编码，未使用旧Dual Anchor或将video首token冒充image anchor。

## 重建检查

Judge完整查看四段224张重建帧，以及工业/暗草地原分辨率source/reconstruction末帧；全8条重建/并排视频逐帧解码/24fps/PTS递增核对PASS。

| 真实scene | 重建前压缩MAD均值 | PSNR逐帧均值dB | 观察 |
| --- | ---: | ---: | --- |
| 43866101 山路 | 5.165 | 29.736 | 主体/树木/地形和运动保持，细节柔化 |
| 7199292c 草地 | 5.889 | 29.469 | 人物/草地运动保留，草纹理有损 |
| 9dc2e588 工业 | 5.792 | 29.251 | 人物完整，道路/工业建筑保留，远处细节柔化 |
| b784d995 暗草地 | 3.563 | 34.894 | 暗部与遮挡来自原片，主体和草地可辨，细节略平滑 |

没有持续结构瓦解或全彩噪。普通VAE有损细节按可行性接受；这些是输入重建，不是合成动作控制通过，PSNR不作为世界模型生成质量指标。

## 决策

真实I0与C1/C2 GT latent可保留用于后续研究。EXP-015发现的变长未来文本影响Global位置问题仍未解决；完整训练入口尚未通过。下一项EXP-017仅CPU设计独立的action-content-independent packed布局候选和回归，不改正式V3代码/权重/结果，不启动GPU或训练。后续实际模型同状态回归与真实mixed-action评估仍须另立预算。
