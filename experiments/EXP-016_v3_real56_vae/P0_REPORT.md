# EXP-016/v1 P0 Worker 报告

2026-10-11 08:43 HKT。**P0 CPU入口已完成；P1 GPU尚未获 marker，实际0 GPU/0模型加载/0 VAE调用/0训练。** 当前实验仅测试EXP-015四段真实56帧在冻结H3 source video VAE中的39/56编码前缀和56帧重建，不包含text encoder、DiT、action packed、AnyFlow或DMD。

## 冻结输入、代码与实际预检

- [任务书](taskbook_v1.md)的四scene顺序与 [source manifest](source_manifest.json)一致；EXP-015父清单SHA为 `adbd82f57f82a17cd0b830192d809b5e524922a91e2ffbca508ac2173775f051`。新MP4/I0逐文件SHA重新检查；四片各56帧、24FPS、832×480，I0逐像素等于第一帧。输入只用同一新56片解码出的PIL帧及其前39切片，旧39压缩片和旧Dual Anchor均不读取。
- source VAE权重：`H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/video_vae/source/model.safetensors`，SHA `5f0c2e161d895a9fee7645ca32d4a7e3a22b90cacfcbeba62ec999cdbbefe0d3`；runtime VAE实现SHA `ff1eea77a85b76a98bbb87c622e7c313996c4348ca14c8e8aa433f27c22ee1d7`。精确源码/配置清单见 [code manifest](code_manifest.json)，后续marker需绑定其SHA和source/config SHA。
- CPU预检命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-016_v3_real56_vae/run_vae.py --preflight`。实际输出见 [P0.log](P0.log) 和 [P0_PREFLIGHT.json](P0_PREFLIGHT.json)：4 CPU affinity、四片224帧、原生 `preprocess_video(..., min_value=0,max_value=1)` 得到 `float32[0,1]`，同一数组前39与前缀输入逐值相同；39→12、56→17的原生时间长度已核对。构造函数默认 `causal_encoder=True`、`use_t_isolated_gn=True`、`clip_length=17`、`token_drop=3`。这是源码/CPU检查，不等于实际权重VAE的数值前缀已验证。
- 无marker调用 `--encode` 时按预期在模型/GPU导入前抛出PermissionError；[拒绝日志](NO_MARKER_REJECTION.log)保留，未创建GPU账本或输出。[run_vae.py](run_vae.py)仅在P1 marker SHA校验通过、GPU0空闲与08:50启动闸门通过后进入模型加载；每次image/video encode和decode前写入调用账本，逐scene输出不覆盖，600秒/44GiB/1GiB/60GiB/08:57闸门已编码。

## P1预定命令与停止条件

只有Judge创建 `judge/P1_APPROVED.json` 并绑定config/source/code SHA后才执行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-016_v3_real56_vae/run_vae.py --encode \
  > submission/experiments/EXP-016_v3_real56_vae/P1.log 2>&1
```

单GPU0顺序处理，每图1 image encode、2 video encode、1 full17 decode，总计4/8/4；0 text/DiT/训练。前缀比较若超过max_abs 0.02或relative_RMS 1e-3，先保存已生成真实tensor和数值再停止后续scene，绝不自动切fp32或重试。重建视频需Judge逐帧看主体/场景；P0没有任何图像质量或编码成功结论。当前等待Judge P1放行。
