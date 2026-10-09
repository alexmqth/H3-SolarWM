# V3 · Efficient Causal H3-World — 73帧候选已验收

正式124帧V3尚未完成；目录保留planned名称以保持链接。

1. **研究目标**：同一配置统一strict chunk-causal、真实persistent KV、可辨动作与可用多窗口结构。
2. **Parent**：Original H3 + released action LoRA；native条件借鉴V2b，复用旧current-prefix候选；没有新checkpoint训练。
3. **核心改动**：private current prefix/own-action反馈、sigma0/native提交的历史raw KV，历史只读复用；不同于V2b Same-σ历史每步双向重算。
4. **配置**：Single I0，12→5→5，30steps/chunk，shift2.22，全局RoPE，停车场832×480，seed13，固定audio，零新增训练。
5. **代表视频**：[AA/AD73帧对比](../../report/V3_native_cached_candidate/README.md)。
6. **量化证据**：第二块同history A/D flow +1.347/−1.458；第三块+0.474/−0.930。124新增forwards、4VAE、0.22923GPU-hours，peak26,686MiB（首条prefill测量缺项披露）。
7. **已经验证**：严格因果/真实缓存与可辨当前动作响应及基本视觉同时存在于同协议73帧候选中。
8. **限制**：AA55→56明显姿态跳变、画质与节奏欠佳；单scene/seed，未有124帧/长期泛化/完整E2E或公平speedup。
9. **下一步**：只延伸同候选73→124和成本测量；若严重持续失效则停止，不自动增加训练。
10. **来源**：[EXP-002](../../experiments/EXP-002_native_cached/README.md) · [正式验收](../../experiments/EXP-002_native_cached/judge/FINAL_REVIEW.md) · [输入与源码凭据](../../experiments/EXP-002_native_cached/MANIFEST.md)。

模型未经大量针对性训练；当前接受可行性质量，普通细节和边界缺陷不构成暗加成熟画质门槛。AnyFlow/DMD仍待可信因果基线后推进。
