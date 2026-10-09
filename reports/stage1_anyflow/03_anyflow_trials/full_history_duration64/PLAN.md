# Full-history AnyFlow训练量对照：16→32→64

只改变optimizer更新总数；从已完成的native-FP32/full-history16 step16恢复完整LoRA、Adam和CPU/CUDA/logical RNG。训练、validation与inference shift都保持2.22；不混入GPU1的shift12分支。使用已冻结的full-history runtime，未修改其代码。

GPU0独占，reserve6GiB；43,237,376个全50+2块rank8 QKVO/FFN参数；旧visual/action/time冻结。39f、3chunks、RGB dual、CPU KV、causal action rows与feedback、same image/prompt/seed13和noise固定。

先到总32更新，保存checkpoint并跑A/D8-step；若数字gate通过即停止进一步训练、等待视觉与少步效果复核；否则精确续训到总64并评测A/D4/8。总计最多48次新增更新和6条39f视频。按原16次约48分钟估算，新增训练约2.4小时，另有验证与评测开销；这是估计。

理由：full-history16的公共weighted loss变化<0.1%，不能把16次smoke当充分训练。此分支测量训练量是否能让同一objective开始学习；不是已知可行的修复。官方H3 Stage1配置max_steps=30000/global_batch128，本分支仍只两条teacher伪真值、logical batch4。GPU1独立的shift12对照用于检查另一项训练分布差异，两个效应不混合归因。

自动数字gate不等于视觉通过。必须A>0、D<0、A−D>1与人物/车库稳定，且有相对匹配FM的少步收益，再独立seed、switching、124f。Stage2暂缓。不覆盖meeting视频。
