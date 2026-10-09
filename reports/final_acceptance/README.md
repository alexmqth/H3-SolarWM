# 最终可复现性验收：工程通过，质量结论不变

2026-10-09按[REPRODUCE.md](../../REPRODUCE.md)完成独立环境验收。**最小推理、KV/因果测试、小H3训练smoke全部通过；E2仍冻结为负结果，没有新增研究训练。** 机器OS/驱动及现有模型权重共享，Python依赖和DiffSynth源码独立重建；这是新venv验收，不是全新OS/容器验证。

## 检查结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 干净Python环境 | 新Python3.10 venv，no system/user site-packages，PYTHONPATH为空，pip check无冲突 | [环境收据](environment.json)、[完整依赖版本](pip_freeze.txt) |
| 干净DiffSynth源码 | 从GitHub克隆，固定`300e3e4…`，按提交包顺序应用5份patch | [源码/依赖预检](runtime_preflight.json) |
| 权重/adapter | base shard索引与键可读；发布H3 LoRA哈希符合，104对全部加载；包内visual tail16/action tail8哈希一致 | [setup.json](inference_A39/setup.json)、[主片来源](../../meeting/DEMO_PROVENANCE.md) |
| KV与局部因果性 | 15 tests passed，含cached/recompute、淘汰、尾块、动态anchor、未来不可见等 | [测试日志](kv_and_causality.log)、[JUnit](kv_and_causality.xml) |
| 训练smoke | 真实MiniMaxH3DiT随机小配置，普通FM，2048可训练参数，8更新，loss2.334223→2.331131，梯度有限且参数改变 | [训练收据](training_smoke.json) |
| 真实33B causal inference | 39帧、832×480、24fps、H264/yuv420p完整解码，3chunks、24noisy forwards＋3clean commits | [新生成视频](inference_A39/cached.mp4)、[运行记录](inference_A39/cached.json) |
| 测量输出 | 耗时、GPU allocated、CPU raw KV、灰度MAD、boundary MAD、Farneback均有输出 | [acceptance.json](inference_A39/acceptance.json) |
| 展示视频 | 主片、四方向grid、20秒失败片与6条E2局部对照共9条完整解码 | [播放格式检查](presentation_video_validation.json) |
| 源码独立性 | 最终提交包code/tests/scripts与独立验收副本哈希一致 | [源码一致性清单](source_consistency.json) |

## 39帧单次执行测量

| 项目 | 实测 |
|---|---:|
| 端到端含shared setup | 374.586 s |
| 采样 | 326.991 s |
| 内部首chunk | 92.430 s |
| 后续两chunk | 119.063 / 115.494 s |
| GPU峰值allocated | 15176.43 MiB / 14.82 GiB |
| CPU raw KV峰值 | 6484.13 MiB / 6.33 GiB |
| noisy / commit forwards | 24 / 3 |
| 相邻灰度frame MAD | 3.5418 |
| A平均水平Farneback flow | −1.14536 px/frame（方向仍未通过） |
| 灰度boundary MAD，名义17/34帧 | 4.2148 / 4.1027 |

使用物理GPU2，其他进程已有负载，`ABOT_VRAM_RESERVE_GIB=20`，一次执行、无warmup。降低模型驻留预算改变offload行为，不能把这次39f显存/耗时与旧124f或其它reserve配置直接排名。没有权重/激活/KV显存组件分解，没有新增124f重复性能基准。灰度MAD与会议历史表的RGB MAD定义不同，不能混算。

这条A视频加载的是主片旧checkpoint，验收脚本不以flow符号或MAD作质量PASS。视频完整可播放、缓存可执行与画质/动作正确是不同结论；它也不代表最新E2。

## 本次修复的提交问题

- 补入遗漏的`tests/test_h3_cached.py`，requirements补齐pytest和OpenCV。
- 新venv原始pip解析`typing_extensions`元数据失败，升级到26.2.1后安装成功；说明中已固定该步骤。
- 新增显式外部权重/首帧/prompt参数，实际记录权重和adapter哈希，检查104对released LoRA完整加载。
- 提供独立源码重建/预检脚本，不依赖原研究源码checkout的未提交修改，不自动下载模型或数据集。
- 缓存目录由env.sh预建，避免新环境的Torch kernel-cache目录缺失警告。该目录修复不改变模型数值或训练协议。

小模型训练smoke不证明33B训练效果；本次真实权重测试是推理。没有运行新的E2 optimizer、AnyFlow或Stage2。完整汇总见[summary.json](summary.json)。
