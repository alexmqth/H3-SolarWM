# EXP-025/v1 Worker 报告：全50层 QKV rank8、C1/C2 普通 FM

状态：`completed`，**待 Judge 验收**。执行日期：2026-10-15 夜间至 2026-10-16 00:03 HKT。单张实际空闲 L40（`CUDA_VISIBLE_DEVICES=0`），没有使用其他 GPU，没有运行 AnyFlow、DMD 或额外训练。完整协议见 [任务书](taskbook_v1.md)和[授权](AUTHORIZATION.json)。

## 结论

全50层 C1/C2 普通 FM 的训练与评测链路真实完成，资源、路由、冻结参数、历史 KV、checkpoint 和媒体审计均通过。**模型能力收益未证实。** 固定 C2 FM probe 的 MSE 在 train/dev 分别下降约 1.42%/1.25%，但 C1 分别上升约 0.21%/0.26%；六组视频没有呈现跨角色一致而明确的画质或动作提升。本轮只检验32条训练 clip、2条已观察 dev clip、56帧两块和8NFE；不证明更长时间或 A/D 同状态动作方向恢复。建议保留为受控联合基线，不升级正式 V3 checkpoint，也不据此自动追加 rank、epoch、AnyFlow 或 DMD。

与 EXP-023 相比，新方案**同时改变** adapter 覆盖（tail8→all50）和训练块覆盖（C2-only→C1+C2），不能把任何变化单独归因于容量或 C1 训练。

## 冻结输入与执行

- Original H3 + released Action LoRA 冻结；全50层新零增量 QKV rank8，10,752,000可训练参数。Single I0、native time、Global RoPE、own-action/current-prefix feedback、strict chunk-causal、GT C1 clean commit→CPU persistent raw KV。无新编码、文本处理或下载。
- 复用 EXP-023 的32 train和2 dev、统一源 RGB 编码的17-latent条件。12+5 latent，C1/C2各128次，32 clip × 每块4遍；每次独立噪声和内部 `(0,1)` sigma，AdamW lr1e-4。训练历史为 GT C1，不是模型自生成 history。
- 先做6次零 adapter 回归；然后checkpoint0固定6片视频及C1/C2 probes；256次更新后checkpoint256同协议重复。两条 self-history 片段前后各自自由生成，故其 C2 并非“同状态”消融。
- [CPU预检](CPU_PREFLIGHT.json)核对34条条件实物的102个 tensor SHA、冻结源码、256顺序配额及预计570次DiT。[CPU checkpoint生命周期](CPU_CHECKPOINT_LIFECYCLE.json)验证50层 adapter、populated AdamW 与 RNG 保存重载。[零增量回归](ZERO_ADAPTER_REGRESSION.json)的三类输出与完整 K/V/RoPE SHA 在 disabled/enabled 状态下逐值一致。
- 首个 [C1](FIRST_C1_GRADIENT_GATE.json) 和 [C2](FIRST_C2_GRADIENT_GATE.json) 真实更新均覆盖50层 forward/backward，C2有50层 clean commit；冻结 base/action LoRA 不获梯度，C2完整历史 KV 在 forward/backward 中未改变。每次后续更新仍检查有限梯度、路由和历史 KV 内容摘要。

实际启动命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-025_v3_fm_full_coverage/run_exp025.py --cpu-only
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 timeout --signal=TERM --kill-after=10s 13770s .venvs/h3world/bin/python -u submission/experiments/EXP-025_v3_fm_full_coverage/run_exp025.py --gpu 0 > submission/experiments/EXP-025_v3_fm_full_coverage/RUN.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-025_v3_fm_full_coverage/finalize_cpu.py > submission/experiments/EXP-025_v3_fm_full_coverage/FINALIZE_CPU.log 2>&1
```

[训练/评测实现](run_exp025.py)和[纯CPU媒体实现](finalize_cpu.py)是本轮新增代码；冻结 EXP-023/024 文件未改。实际命令、逐调用与资源见 [RUN.log](RUN.log)、[GPU_LEDGER.json](GPU_LEDGER.json)、[逐更新记录](TRAIN_UPDATES.json)。checkpoint 0/64/128/256 含 adapter、optimizer、CPU/CUDA/noise RNG与训练顺序；大型 `.pt` 保留本地，对应 `.json` 保存 SHA 和大小。

## 资源与固定数值

| 项目 | 实际 | 授权上限 |
| --- | ---: | ---: |
| 顶层 DiT 尝试 | 570 | 576 |
| backward / optimizer 更新 | 256 / 256 | 256 / 256 |
| VAE decode / encode / text encode | 12 / 0 / 0 | 12 / 0 / 0 |
| GPU 进程总时间 | 11,566.6 s | 13,800 s |
| 回归 / 前评测 / 训练 / 后评测 | 76.1 / 781.6 / 9,666.1 / 1,042.8 s | 400 / 1,400 / 11,000 / 1,400 s |
| 峰值 GPU allocated / reserved | 24.45 / 28.14 GiB | allocated <44 GiB |
| 单块 CPU raw KV | 6,799,104,000 B | — |

固定 velocity probe（MSE；每格为两条 clip × sigma 0.25/0.75，共4项；负变化为下降）：

| Split / chunk | checkpoint0 | checkpoint256 | 相对变化 |
| --- | ---: | ---: | ---: |
| train C1 | 0.140055 | 0.140352 | +0.21% |
| train C2 | 0.162249 | 0.159937 | −1.42% |
| dev C1 | 0.134542 | 0.134886 | +0.26% |
| dev C2 | 0.146192 | 0.144358 | −1.25% |

视频连续性 proxy（均值；GT-history n=4，self-history n=2）：

| Protocol | boundary MAD 0→256 | frame MAD 0→256 | 解读 |
| --- | ---: | ---: | --- |
| GT39 + generated17 | 10.49→10.85 | 10.37→10.27 | 边界未改善 |
| self-generated C1+C2 | 12.41→9.72 | 9.56→8.96 | proxy下降，但两条自生成轨迹不同，运动减弱也可造成下降 |

逐项值见 [METRICS_SUMMARY.json](METRICS_SUMMARY.json)、[前评测](EVAL_PRE_SUMMARY.json)与[后评测](EVAL_POST_SUMMARY.json)。随机训练 loss 不构成固定学习曲线；此处没有 A/D 同状态 counterfactual，不能报告动作方向正确率。

## 视频检查

[六组三列对照](comparisons/MEDIA_AUDIT.json)以左 GT source / 中零 adapter / 右256-update adapter 编排，每条56帧、24 FPS；原片分别在 [eval_before](eval_before) 与 [eval_after](eval_after)。可优先看 [train_switch GT-history](comparisons/train_switch_GT39_generated17_GT_zero_256.mp4)、[dev_constant self-history](comparisons/dev_constant_self56_GT_zero_256.mp4)、[dev_switch self-history](comparisons/dev_switch_self56_GT_zero_256.mp4)。各条另有六时点 contact sheet 和逐帧小图。

六组原片的前后 clip、condition、seed、初始噪声均相同；四组 GT-history 的前39张显示 RGB 逐值相同。[独立视频审计](INDEPENDENT_VIDEO_AUDIT.json)再次完整解码12条原片；[媒体审计](comparisons/MEDIA_AUDIT.json)记录六条对照片的解码帧数、FPS和SHA。全56帧缩略索引及关键帧显示：人物在两条 self-history dev clip 中总体保持可辨，没有新的瞬间整场景切换；与零 adapter 相比，256-update画面主要是位置和细节的微小变化，未出现跨角色一致的明确修复。GT-history片段也没有可靠的视觉优势。参考 GT 的具体未来本身不是模型唯一正确轨迹；位置差异不能单独解释为动作控制改善或退化。

当前最有价值的判断是：全层参数和 C1/C2 覆盖已经不再是“没训到”的工程借口；在这个小数据、普通FM、8NFE协议下，它仍没有提供足够清晰的能力收益。32条 train/2条已观察 dev 样本不足以否定更大规模的FM训练或其他目标，但本轮结果不支持无条件追加同类预算。
