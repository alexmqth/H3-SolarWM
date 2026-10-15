# EXP-020：Original H3 真实混合动作参考

按 [Judge 冻结任务书](taskbook_v1.md)仅运行一个固定训练 episode `b784d995`：原始首帧、原 scene caption、17 个真实动作句（前13个左平移、第14个静止、后3个前进），seed13、56 RGB帧、30 NFE。原版双向 H3 pipeline 和 released step-10000 Action LoRA；`causal_chunk_size=None`，无新增 adapter、AnyFlow/DMD 或训练。

本实验的 GT 视频只是参考，模型从 I0 独立生成全56帧。它与 EXP-019 的“真实前39帧 GT-history，只新生成17帧”**输入、可见历史和推理范围不同**；同一场景不能使二者成为严格单变量因果拓扑消融，也不能直接比较整片 wall 作为同口径加速比。

CPU预检记录[来源、代码和输入 SHA](SOURCE_MANIFEST.json)、[协议核查](CPU_PREFLIGHT.json)与[启动环境](LAUNCH_ENV.json)。冻结 wrapper 为[run.py](run.py)，使用原版 `MiniMaxH3Pipeline.__call__`；实际30 DiT forward、4 text encode、1 image encode、1 video decode、1 audio decode，总 wall **282.237秒**、峰 allocated **26.049 GiB**，零训练。[完整 Worker 报告](WORKER_REPORT.md)和[CPU媒体审计](CPU_AUDIT.json)分别给出逐帧质量判断与56帧/24fps/PTS/SHA核查。

可直接播放：[Original 56帧](artifacts/b784d995/original56.mp4) · [GT vs Original 标注并排](artifacts/b784d995/gt_vs_original56.mp4) · [末17帧逐帧图](artifacts/b784d995/all_late17_contact.jpg)。人物结构大体可辨，但约RGB20以后天空出现放射状结构伪影；左移→静止→前进的动作切换缺少可信反事实验收。模型执行成功不等于画质和动作能力正式通过。原始输出、日志和比较片副本见[SHA清单](ARTIFACT_MANIFEST.json)；Judge最终验收：工程PASS、质量/动作PARTIAL。

## Judge 最终结论

任务已accepted并收口；[最终审核](judge/FINAL_REVIEW.md) · [独立来源/媒体/预算核查](judge/FINAL_AUDIT.json)。执行完成与能力边界分别记录，无新增GPU授权。
