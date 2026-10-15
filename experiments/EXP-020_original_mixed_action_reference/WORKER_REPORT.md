# EXP-020/v1 · Original H3 真实混合动作参考 Worker 报告

2026-10-15 HKT。按 [冻结任务书](taskbook_v1.md)只运行 `b784d995` 一个固定 train episode：真实 I0、原 scene caption、17 个真实动作句，前13个左平移、第14个静止、后3个前进。Original H3 + released step-10000 Action LoRA 在原生双向 `MiniMaxH3Pipeline.__call__` 中，从 I0 生成完整 **56 RGB帧**。30 NFE、seed13、832×480/24fps、CFG1、video flow shift2.22、audio shift3.0；未启用 causal mask、persistent KV、新布局、AnyFlow 或训练。这是一个原版参考样本，不代表 Original 的多场景动作验收。

## 协议与预算

- CPU 预检核对了动作脚本 `13左移→1静止→3前进`、首图和 caption 来源、release LoRA SHA、冻结 infer/pipeline/DiT 源码、原生双向调用及联合音视频 scheduler/decoder。[来源清单](SOURCE_MANIFEST.json)、[CPU预检](CPU_PREFLIGHT.json)、[GPU0启动环境](LAUNCH_ENV.json)保存了实物与 SHA。可编辑 DiffSynth 导入解析到 `/home/qma/work/GWM/H3-World/DiffSynth-Studio-h3-v2`；已核对其 DiT 与 pipeline 两文件逐字节等于冻结 runtime 副本，实际来源写入清单。
- 唯一一次 GPU0 运行使用 `CUDA_VISIBLE_DEVICES=0 ABOT_VRAM_RESERVE_GIB=18 OMP_NUM_THREADS=4`、CPU affinity40–43、外层1200秒 watchdog。实际 **30 DiT forward、4 text encoder 调用、1首帧 VAE encode、1视频 VAE decode、1音频 VAE decode，0训练**；wrapper 在每次调用前预记账。完整 wall **282.237秒**，峰 PyTorch allocated **26.049 GiB**、reserved **26.840 GiB**，均未达到上限。
- 运行后的 shell 尾部用了 zsh 只读变量名 `status`，因此外层 shell 返回码为1；**模型进程自身正常写出 `COMPLETE_PENDING_JUDGE`**，56帧 MP4、计数及完整 CPU 解码审计通过。这个 shell 收尾错误没有触发模型重试或第二次 GPU 调用。原日志保留于[RUN.log](RUN.log)。

## 视频与逐帧观察

- [Original 全56帧](artifacts/b784d995/original56.mp4)、[GT与Original并排标注片](artifacts/b784d995/gt_vs_original56.mp4)、[RGB39–55全部17帧逐帧图](artifacts/b784d995/all_late17_contact.jpg)和[全片抽样图](artifacts/b784d995/keyframes_contact.jpg)均来自此次真实输出。Original 视频包含音频流；并排片只保留视频。三条56帧视频均为24fps，分辨率、PTS、SHA和完整解码已在[CPU审计](CPU_AUDIT.json)核查。
- Original 的人物在全片大体保持一个可辨身体，草地空间连续，没有持续人体分解。约 RGB20 后天空出现明显放射状云纹/结构伪影，GT 没有同等形态；这是可见画质缺陷，不应称原版该样本“完整保真”。
- 早段人物有持续步行、背景移动。RGB43附近对应静止、RGB47附近对应前进是按原生 action span 与 VAE 时间网格的**近似显示区间**；人物在后段仍有步态变化，但仅凭一条生成视频看不出可信的“左移→静止→前进”分段响应。中央背景光流分段和 GT 量级不同，且受整片不同生成背景影响，不能当角色动作正确率。缺少同一生成状态下的动作反事实，本样本**动作切换未通过正式验收**。

## 与 EXP-019 的关系

EXP-019 `b784d995` 是**真实前39帧 GT history + 新生成17帧**，两条普通 FM30/FM8 使用 causal raw KV；本 EXP-020 是 Original 双向模型**从 I0 生成全56帧**。虽然初图、scene、脚本和 seed 对齐，实际历史状态、噪声形状、音视频联合路径和生成范围不等；两组视频是能力/质量参考，不是单因素拓扑消融，也不能用282.237秒与 EXP-019 C2 局部采样段直接计算因果模型的端到端加速比。

原片、并排片、图、结果账本及日志有逐字节副本和[SHA清单](ARTIFACT_MANIFEST.json)。Worker 结论为：**原版全56帧生成工程通过、人物结构大体可辨，但该单场景天空有结构伪影，动作切换正确性尚无充分证据。** 该结果应由 Judge 最终审核后决定如何进入正式版本/汇报材料。
