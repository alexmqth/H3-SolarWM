# V3：正式基线与两条研究路线

正式版本仍为[V3 Baseline](v3_baseline/README.md)：Original H3 + released Action LoRA、严格因果、持久KV、Global RoPE，30步124帧有限可行。

- [Sliding Window](sliding_windows/README.md)：[Global](sliding_windows/global/README.md) / [Local](sliding_windows/local/README.md)，解决历史KV容量增长与视频位置协议问题。
- [FM → AnyFlow](fm2anyflow/README.md)：普通Flow Matching减步和独立AnyFlow学生训练；FM8为每块8步普通采样。
- [V4 SGF / DMD](../v4/README.md)：DMD按研究路线归V4，当前配置生成失败。

mainline保存版本定义和结论，实际可播放视频统一在[对应报告](../../report/v3/README.md)的各`videos/`中。[常用缩写](../../report/GLOSSARY.md)。
