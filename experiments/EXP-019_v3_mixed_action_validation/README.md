# EXP-019：固定动作布局真实模型验证

本轮承接EXP-017，通过真实H3-World检查动作文本变长时的布局/位置和缓存协议，再比较真实历史条件下的普通Flow Matching（FM）续写。FM-30NFE/FM-8NFE表示每个新chunk分别调用网络30/8次；没有AnyFlow训练。

## 当前结果（2026-10-15）

- R已验收：4组velocity、50层150组K/V/RoPE逐值一致，10forward，379.384秒。
- E已验收：四真实I0/head与混合动作编码，14text，192.620秒；未来embedding隔离通过。
- G1已验收：首scene局部视觉可用，动作语义PARTIAL，39forward/332.172秒。仅后17帧新生成；[GT/FM30/FM8对比视频](artifacts/G1_43866101/source_vs_fm30_vs_fm8.mp4)。
- [G1审核与G2放行](judge/G1_REVIEW_AND_G2_RELEASE.md)：其余三scene按固定顺序单卡执行，仍无训练。
- [R审核](judge/R_REVIEW_AND_E_RELEASE.md) · [E审核及G1授权](judge/E_REVIEW_AND_G1_RELEASE.md)。
- 大型fixture `.pt` 仅本地保留，Git发布来源清单、哈希、代码、日志和报告；manifest中的本地tensor并非Git已包含文件。

## 执行阶段

1. CPU来源与输入检查。
2. canonical A旧/新布局的C1/C2数值回归与clean KV比较。
3. 新I0/原caption与真实混合动作独立文本编码。
4. 固定四scene真实C1历史下的C2续写，首scene审核后才扩展。

任务、资源与停止条件见[任务书](taskbook_v1.md)。源码[run.py](run.py)为独立入口，冻结V3不修改。阶段结果会归档至artifacts/，Judge审核位于judge/。当前状态以根目录progress.md和next_plan.md为准。

当前执行记录：P0 已完成；[R canonical 回归](R_WORKER_REPORT.md)与[E 真实条件编码](E_WORKER_REPORT.md)均已由 Judge 审核接受。四场景 E fixture、[独立 CPU 审计](E_CPU_AUDIT.json)和[原始证据 SHA 清单](E_ARTIFACT_MANIFEST.json)均已保存。[G1 首场景 GT-history 续写](G1_WORKER_REPORT.md)已完成并提交待 Judge 质量审核，[56 帧三列视频](artifacts/G1_43866101/source_vs_fm30_vs_fm8.mp4)与[全部 17 张新增帧](G1_ALL_NEW17_CONTACT.jpg)可直接查看。G2 余三场景尚无 marker，未启动。

## 视频口径

本轮是GT-history（真实视频历史）局部续写，前39帧展示真实录屏，仅后17帧是生成。原始decoder输出与拼接展示分别保存；原视频是参考，不要求随机生成逐像素复现真实未来。四段来自train episodes，不宣称验证集泛化、长时闭环或完整D反转。
