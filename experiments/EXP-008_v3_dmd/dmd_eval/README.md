# cycle8 DMD8 视频评估：已停止的负结果

最终cycle8评估使用FM8共同首39帧、相同seed13/噪声与8NFE。AA第二块完整运行至56帧，从第39帧起新增17帧均为全画面彩色噪声；[三列对比视频](artifacts/failure_comparison/AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4)和[全部新增帧](artifacts/raw/AA/chunk_12_17_all_frames.jpg)直观显示失败。视频本身为可解码H.264，旧39帧raw RGB未改。

[端点诊断](failure_latent_diagnostic.json)显示DMD8 C2终点与初噪声cosine 0.949，FM8/AF8约0.100/0.114，支持未有效去噪。该指标只定位结果，不能独立证明具体训练错误。

AD第二块在停止指令交叉时被SIGINT中断，完成5/8个sampling step，第6次调用已计费，没有视频；AA/AD第三块均未执行。累计实际账本16forward、1VAE、205.104 GPU0秒。原始budget、日志、AA原片、失败AD JSON在[artifacts/raw/](artifacts/raw/)；[manifest](failure_evidence_manifest.json)列出SHA与外部cycle8 checkpoint。[完整Worker报告](../worker_report_v2_eval.md)和[Judge最终审核](../judge/FINAL_REVIEW.md)说明执行边界。

本目录的[原计划73帧比较脚本](make_comparison.py)只作历史准备，不应在当前cycle8配置上启动缺失阶段。实际交付由[56帧负结果比较脚本](make_failure_comparison.py)生成。
