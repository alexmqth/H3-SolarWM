# V4 · Efficient Causal H3-World — Planned

1. **Version Name / Research Objective**：将V3可信局部动作与视觉能力迁移到高效因果骨干。
2. **Parent Version / Baseline**：V3推理协议与Original权重正控；不是已完成的新checkpoint。
3. **Main Changes**：计划研究strict chunk-causal attention、明确的public-prefix规则、原生action路由和历史时间、persistent KV。
4. **Model and Inference Configuration**：待定；先30-step可信causal，再AnyFlow，最后on-policy DMD。
5. **Representative Videos**：**NOT_YET_VALIDATED / 无视频**，不从V3复制冒充V4。
6. **Quantitative Results**：无模型性能结果。
7. **What Was Improved**：已有机制审计给出了需要控制的变量，不等于V4能力通过。
8. **What Still Failed**：strict causal + persistent KV尚无动作/画面联合PASS。
9. **Lessons Learned**：先证明相同计算图下cache/recompute等价，再评判局部动作；cosine不等于质量。
10. **Source / Checkpoint / References**：[下一步验收](../../report/02_next_steps/README.md)；**无checkpoint**。

本次仅整理材料；不自动启动任何下一步实验。
