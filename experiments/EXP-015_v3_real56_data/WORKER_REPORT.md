# EXP-015/v1 Worker 报告

2026-10-11 HKT。当前 [任务书](taskbook_v1.md)只授权 CPU 数据准备。结果：**四段真实 56 帧/17 latent 的来源、时间、动作与文件完整性检查通过；0 GPU 调用、0 模型编码、0 前向/反向、0 训练更新。** 这不是普通 FM、AnyFlow 或 Stage2 的训练结果。具体 scene/action 子目录与文件入口见 [README](README.md)。

## 冻结输入与实际执行

- [config](config.json)固定 EXP-013 `candidate_manifest.json` SHA `b6af2b0208a5dcf83c02f76fa826e5a03ba52fa9bd83de357f9474b5efb257b6` 的四个 train episode/source start：`43866101/1245`、`7199292c/1670`、`9dc2e588/1410`、`b784d995/1635`。源视频与annotation原路径见 [EXP-013来源清单](../EXP-013_v3_multiscene_data_plan/source_manifest.json)；SHA、静态caption、split与旧39帧的关系见本轮[输出manifest](OUTPUT_MANIFEST.json)；没有换片、扩场景或下载。
- 执行入口：[build_real56.py](build_real56.py)、[audit_alignment.py](audit_alignment.py)、[audit_filter_pts.py](audit_filter_pts.py)。命令均设 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`，使用 `.venvs/h3world/bin/python`。先执行 `build_real56.py --preflight`，再 `--build`；后执行 `audit_filter_pts.py`。实际日志：[预检](preflight.log)、[首次构建与中断](build.log)、[续建](build_resume.log)、[源 PTS](source_pts.log)、[像素抽查](alignment.log)。

在项目根目录运行的实际入口形式如下；原始日志包含首次失败和续建细节，不能只看最后一条成功命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-015_v3_real56_data/build_real56.py --preflight
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-015_v3_real56_data/build_real56.py --build
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-015_v3_real56_data/audit_alignment.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-015_v3_real56_data/audit_filter_pts.py
```
- 首次构建在第一条 FFmpeg filter 错误处失败，0产物；修正 filter 后前三段生成并检查通过。观察到 FFmpeg 自动线程数超出任务书的4线程上限，因此在第四段前主动中断。续建不覆盖前三段的 MP4/数组，只逐值复核它们；第四段显式设解码/滤镜/编码各1线程。系统监测的 FFmpeg OS 线程峰值仍为首次9、续建5，故**不能声称严格满足≤4 OS线程限制**。后续源 PTS 审计固定4 CPU affinity。没有因像素 MAD 重建任何片段。
- 四段新增文件总计 `8,412,751` bytes，远低于1 GiB；当时磁盘剩余约121 GiB。manifest 的 `wall_seconds=15.8803` 与 `new_bytes=2,174,976` 只统计最后一次续建进程，**不是两次构建累计值**。实际完整产物大小以各文件 SHA/bytes 求和为准。08:50 前完成新增处理。

## 数据和动作核查

| 场景 | I0 对旧39 PNG 的 MAD | 新56前39对旧39视频平均MAD | 原39动作前缀最大误差 | 真实按键 |
| --- | ---: | ---: | ---: | --- |
| 43866101 | 0.281 | 1.246 | 0 | A+S+L，全56帧 |
| 7199292c | 0.031 | 1.373 | 0 | A+S，全56帧 |
| 9dc2e588 | 0.551 | 1.874 | 0 | A+S+J，全56帧 |
| b784d995 | 0.304 | 1.012 | 0 | A 40帧→静止8帧→W 8帧 |

四个MP4各56帧、24FPS、832×480、PTS严格递增且唯一，合计224帧全部解码。源30FPS的第 `j` 个输出帧应来自 `src_start + 5*(j//4) + j%4`；[FFmpeg showinfo 源 PTS 核查](SOURCE_PTS_AUDIT.json)对每段56个实际选中帧逐个验证此索引。所有11个原始key逐行等于源annotation相应帧；17列连续量沿用原COLMAP/episode尺度转换，旧39动作前缀完全相同。Judge的[独立核查](judge/INDEPENDENT_DATA_AUDIT.json)复算224行、17 spans、脚本及C1扰动隔离，均通过。

H3时间宽度是 `(1,4,4,4,4)` 周期。完整56帧对应17 spans，首12恰到RGB39；C2五个span为 `[39,43) [43,47) [47,51) [51,52) [52,56)`。key按span取max；连续旋转/位移按span求和再归一化/裁剪，独立手算最大误差在1e-6内。前12 pooled与旧39数据逐值相同。把C2原始动作清零后，C1 pooled、bounded KEYS9和文字脚本均不变；`bounded_keys9(..., stops=(12,17))` 没有借用未来C2短span。这里的未来隔离只针对**动作预处理**，不代替模型 attention/KV 审计。

真实动作保留A+S/J/L组合；四段Q/E/Space计数均0。未出现同span相反key需要净化。`KEYS9`第9位 F 由J/L与COLMAP yaw速率派生，s2共有11个F=1的latent span；它是观测代理变量，不是用户真实按键或Space。`action_script17.json` 只保留离散动作和相机速度档位，连续位移/旋转幅值仍以 `raw56.npy`、`pooled17.npy` 保存，不能把文字当无损标签。s3的C2有 A→静止→W，不能笼统写成持续A。四段没有真实D反事实，尤其不能把EXP-014合成teacher的纯D视频称为真实GT。

## 源像素抽查的局限

[近邻像素记录](SOURCE_ALIGNMENT_PIXEL_SPOTCHECK.json)用PyAV RGB缩放近似FFmpeg的YUV缩放/重编码，仅为辅助诊断。s0/s2/s3抽查位置预测索引最相似；s1前三个位置的邻帧略低MAD，与EXP-013早期抽查的同场景歧义一致。由于变换路径和低运动内容，这项近似像素排序不能独立证明错位。实际select后、setpts前的[56帧源PTS审计](SOURCE_PTS_AUDIT.json)和前39旧视频MAD、源/标注SHA共同支撑时间对齐。未因该歧义重新编码或修改原始数据。

## 后续编码与训练设计（本轮没有执行）

1. **VAE前缀因果性闸门。** H3实际 `MiniMaxH3VideoVAE` 默认 `causal_encoder=True`、`use_t_isolated_gn=True`、`clip_length=17`、`token_drop=3`；`encode_temporal`按17 RGB分片，尾部pad并去掉末3个token。按相同新56帧视频和实际dtype/runtime，分别编码完整56→17及它的前39→12，比较前12 latent的逐值/相对误差，再决定是否允许以全56编码作为C1监督。不能拿旧39重编码MP4来混淆这个测试；当前未做GPU VAE编码，也未证明C1 video latent无未来依赖。
2. **原生条件。** 新 `I0.png` 经H3 `process_image=True` 形成Single I0；视频通过 `process_image=False` 独立编码；静态caption取原annotation，17个真实动作经固定 `bounded_keys9(stops=(12,17))` 和 `action_script17` 生成token。保留native timestep、global RoPE及严格的当前动作可见范围，不复用旧Dual Anchor fixture。
3. **分清三类历史。** 真实C1 teacher forcing用观察到的C1视频/动作作条件、真实C2作目标；合成teacher C1是模型生成的39帧条件，与真实GT不同；student自生成C1则是on-policy状态。若student参数更新，其自己的C1/KV必须用更新后参数重建，不能复用旧KV。需要分别记录 GT-history 与 generated-history 结果。
4. **目标和覆盖。** 普通FM直接监督真实C2 flow velocity；AnyFlow的target-time finite-map目标是后续独立实验，不能把普通FM8步称为已完成AnyFlow。DMD只在可信Causal/少步前置验证后讨论。四个train片段且主要含A，不提供D真实后果；这批数据可验证有来源的局部真实转移，不能单靠它声称恢复A/D双向控制或跨场景泛化。
5. **Packed位置的独立CPU闸门。** H3源码 `H3-World/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py` 的 `PackedSequenceBuilder._build_packed_fl2va` 用总`text_len`设置action origin及anchor/video的Global位置。真实动作script句长可变；如果先按全17个action构建，再只裁成C1可见12个，未来C2的文字长度仍可能改变C1位置。后续须分别构造不同C2文本而相同C1的packed输入，检查C1所有可见位置、mask、token与当前条件不变；不可悄悄padding、改prefix或time规则来绕过检查。EXP-015只验证了动作预处理的未来隔离，**没有通过这一packed位置闸门**。

任务到此停止，等待Judge验收和下一项独立授权；没有启动新GPU实验。
