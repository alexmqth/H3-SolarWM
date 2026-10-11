# EXP-017/v1 Worker报告

2026-10-11 08:54 HKT。**CPU候选实现和指定的回归/未来隔离检查通过；0 GPU、0模型forward、0 encode/decode、0训练。** 执行命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-017_v3_fixed_action_layout/fixed_layout.py \
  > submission/experiments/EXP-017_v3_fixed_action_layout/CPU_RUN.log 2>&1
```

使用4 CPU affinity、四个EXP-015冻结action脚本、四个EXP-014同场景full37 A/D fixture和本地H3 tokenizer。冻结builder/`visible_inputs`的实际源码经AST原样提取，SHA见[CPU_RESULTS.json](CPU_RESULTS.json)。没有下载或加载33B模型。CPU运行日志见[CPU_RUN.log](CPU_RUN.log)，候选字段映射与后续GPU闸门见[PROTOCOL.md](PROTOCOL.md)。

| 场景 | 真实前17句token长度 | full37 canonical回归 | stop12/17未来内容+长度隔离 | 已提交history坐标 |
| --- | --- | --- | --- | --- |
| 43866101 | 14×17 | 所有packed字段、A/D可见输入逐值相等 | PASS | 不变 |
| 7199292c | 13×17 | 同上 | PASS | 不变 |
| 9dc2e588 | 14×17 | 同上 | PASS | 不变 |
| b784d995 | 10×13，8×4 | 同上 | PASS | 不变 |

未来干预使用与原句明显不同的内容，并同时增/减token长度；还把未来25句全部缩为单token作边界压力测试。比较的是完整可见token ID、位置、`img_pos/text_pos/audio_pos`、`action_text_rows`、`token_tags`、`cu_seqlens`和prompt IDs，不只是shape或cosine。动作跨度/索引无越界和重复，未来物理行经原`visible_inputs`删除。第一次实现发现原builder的action-aware origin检查仍依赖未来总长，现已改为仅复用其物理行构造并显式设置固定语义坐标；旧日志保留供追溯。CPU结论是**布局协议候选工程成立**；真实混合句子的text encoder/Token Refiner/DiT forward、真实backend mask、raw KV数值等价与生成视频均 `NOT_TESTED`，不能把此结果写成动作控制或训练能力通过。[Judge最终验收](judge/FINAL_REVIEW.md)已接受CPU候选，下一GPU草案未授权，本任务不自行启动GPU。
