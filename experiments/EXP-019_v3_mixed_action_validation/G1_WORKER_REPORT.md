# EXP-019/v1 · G1 首场景 GT-history 续写 Worker 报告

2026-10-15 HKT。Judge 接受 E 后，仅授权 scene `43866101` 的 G1。输入为该真实录屏的前 **39 帧 GT history**（12 latent），先从这段真实历史做一次 clean KV commit；然后使用同一新 E fixture、同一 C2 初噪声和脚本动作分别生成 **17 帧**，FM-30NFE 与普通 FM-8NFE。两段展示视频均为 56 帧：前39帧显示真实录屏，**只有 RGB39–55 是模型新生成**。这不是从零生成 56 帧，也不是 free-running rollout。

## 执行、协议和资源

- 冻结 `run.py` 的启动命令为 `PATH=/home/qma/work/GWM/H3-World/outputs/EXP-019_v3_mixed_action_validation/bin:$PATH CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-019_v3_mixed_action_validation/run.py --stage 43866101 --gpu 0`。任务 bin 内 `ffmpeg` 的实际路径、SHA-256、PATH 首位及 marker/source SHA 见 [G1_LAUNCH_ENV.json](G1_LAUNCH_ENV.json)。未改冻结入口或模型协议。
- 单卡 NVIDIA L40 GPU0，UUID `GPU-d77c5635-02ec-c2d2-d744-1dfcc4a14e8f`，启动空闲 45,458 MiB。实际 **39 次 DiT forward = 1 次 C1 clean commit + 30 次 FM30 采样 + 8 次 FM8 采样**；2 次 VAE decode、0 text encoder、0 训练更新。阶段完整 wall **332.172 秒**（上限 900 秒），watchdog 未触发。PyTorch allocated 峰 **24.945 GiB**，reserved 峰 **27.154 GiB**。一次 C1 CPU raw KV 为 **6,799,104,000 bytes = 6.332 GiB**，50 层均只含 index0。
- FM30/FM8 的 C2 初噪声 SHA 均为 `9571b2309b5e8a593acc49c88231a1e8b6f0ecadb8680023bd750308a8dcab74`；均用同一 GT latent history、真实动作句、native scheduler/shift2.22 和冻结 cache。两个 sampling loop 实测分别 **163.322 秒**和 **46.157 秒**；这只是各自采样段耗时，不能当作包含编码/commit/解码/保存的完整端到端加速比。运行中旧 history latent 和 C1 cache SHA 保持不变。

| 续写 | 新生成帧 | 采样 NFE | 采样段秒 | VAE decode 秒 | 边界 MAD* | 段内 MAD* | 中心水平光流和* |
|---|---:|---:|---:|---:|---:|---:|---:|
| 真实后续 GT | 17 | — | — | — | 8.547 | 9.438 | −28.138 |
| FM-30NFE | 17 | 30 | 163.322 | 6.119 | 7.909 | 8.487 | −35.168 |
| FM-8NFE | 17 | 8 | 46.157 | 5.163 | 8.146 | 7.603 | −35.108 |

`*` 三行均由**完整解码后的 56 帧 MP4**用 EXP-014 相同中央区域 Farneback 参数重新计算，保证相同口径；光流只是运动辅助代理，不是动作正确率。原 runner 在压制 H.264 前的 RGB 数组给出的边界 MAD 为 FM30 `8.099`、FM8 `8.312`，与上表解码后数值不能混用。展示 MP4 的前39帧由源 RGB 数组覆盖后重新 H.264 编码，解码后与源 MP4 有约 2.00 灰度级平均像素差，**不宣称压缩后逐像素完全相同**；源码中的 `displayed_prefix_exact_to_decoded_source=True` 指编码前数组。

## 视频与逐帧判读

- [GT / FM30 / FM8 三列 56 帧标注视频](artifacts/G1_43866101/source_vs_fm30_vs_fm8.mp4)将第 39 帧起明确标记为 `GENERATED`；[17 张新帧逐帧三列图](G1_ALL_NEW17_CONTACT.jpg)显示 RGB39–55 全部帧。单独看 [FM30 展示片](artifacts/G1_43866101/fm30_gt39_plus_generated17.mp4)和 [FM8 展示片](artifacts/G1_43866101/fm8_gt39_plus_generated17.mp4)。未覆盖 GT 的原始 VAE 解码片分别为 [FM30 raw](artifacts/G1_43866101/fm30_raw_decode56.mp4)、[FM8 raw](artifacts/G1_43866101/fm8_raw_decode56.mp4)。
- 人物在两条新增片中始终是单一可辨身体，四肢没有持续分解；山路、松树、岩石保持可辨，39 帧边界没有明显瞬间换场。FM8 边缘和纹理稍软，两条均有普通局部模糊，但本场景未见持续透明重影或彩噪崩溃。接触图和原分辨率 RGB39/43/47/51/55 已复看。
- 这段真实脚本在 s0 的 17 个动作 span 是同一句复合动作（后退、左平移、相机缓慢右移），**不是段内 A→D 切换**。两条生成视频都显示持续步行与背景位移；中心区域水平光流代理与真实后续同为负值且量级接近。然而生成的人物较快转为侧向行走，路径和姿态与真实后续有明显差别。仅凭这一条脚本与一个历史状态，不能断言角色的“后退+左平移”语义被正确执行，更不能证明动作反事实或即时切换能力。

## 实施核查与限制

CPU 审计完整解码真实源、两条展示片、两条 raw decode 片和三列片，共 6 个各 56 帧、24fps，尺寸/PTS/文件 SHA 均通过；两端点 tensor 为有限值 `[1,24,5,30,52]`。审计脚本和机器结果见 [audit_g1.py](audit_g1.py)、[G1_CPU_AUDIT.json](G1_CPU_AUDIT.json)，阶段账本、cache hash、原始结果、视频、日志逐字节副本见 [G1_ARTIFACT_MANIFEST.json](G1_ARTIFACT_MANIFEST.json) 与 `artifacts/G1_43866101/`。源码在推理前读取了完整真实源视频供后续展示比较，但未来 RGB 并未传入 DiT forward；实际模型输入为 GT `prefix12.pt`、当前噪声、E fixture 条件与 C1 KV。

Worker 结论是：**G1 的局部视觉续写可用、少步仍能形成可辨运动，动作语义只得到有限且非反事实的支持**。该训练 episode 的 GT-history 结果不支持 free-running、其他场景泛化、D 反转或长期稳定声明。余三场景 G2 没有本 Worker 自行启动，等待 Judge 对 G1 的视觉和资源证据审核。
