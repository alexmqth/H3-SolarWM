# EXP-016 — 真实视频VAE编码与前缀验证

Judge已验收：四scene的同源56/39编码前12latent逐值相等，224帧完整重建可辨。4image/8video encode/4decode，77.721815GPU秒，峰7.503959GiB，0训练。这是数据编码验证，不是生成视频能力或完整训练fixture通过。

[最终Judge审核](judge/FINAL_REVIEW.md) · [Worker报告与四组左源/右重建视频](WORKER_REPORT.md) · [实物tensor/视频审计](judge/INDEPENDENT_RESULT_AUDIT.json) · [调用账本](artifacts/budget.json) · [归档SHA](ARTIFACT_MANIFEST.json) · [P0实现与命令](P0_REPORT.md)

实际模型source VAE、bf16计算、float32规范化latent输出。SingleI0独立process_image=True；视频17latent及39前缀12latent独立编码。没有新增text/DiT/AnyFlow/DMD。变长真实动作的packed位置未来隔离由下一项EXP-017独立CPU候选处理，正式V3结果保持冻结。
