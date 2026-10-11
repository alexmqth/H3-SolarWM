# EXP-016/v1 Worker最终报告：真实56帧 source VAE

2026-10-11 08:47 HKT。**四个固定场景的 native VAE 前缀编码与56帧roundtrip完成；四组同源39/56编码的前12latent逐值相同。** 这证明冻结source VAE在本次真实输入、bfloat16计算与tile配置下没有观察到C2 RGB影响C1 latent。它只验证数据编码/重建，不证明真实动作packed条件无未来泄漏，也不是模型生成、FM/AnyFlow训练或DMD质量结果。

## 协议与来源

P0固定输入与无marker拒绝见 [P0报告](P0_REPORT.md)；Judge [P1 marker](judge/P1_APPROVED.json)绑定 [配置](config.json)、[来源](source_manifest.json)和[源码](code_manifest.json) SHA。GPU0启动时空闲；实际只加载 `FL2VA/video_vae/source/model.safetensors`，不加载text encoder、DiT或audio VAE。每场景使用新56RGB视频本身的前39帧作为短输入，`preprocess_video(..., min_value=0,max_value=1)`产生float32[0,1]；新I0经原生image分支，video经原生video分支，内部VAE计算bfloat16、tile256/overlap64。输出规范化latent实际是float32，按真实dtype保留。33个encoder时间卷积均causal，25个时域隔离norm；`clip_length=17`、`token_drop=3`。

P1唯一命令（原日志 [P1.log](P1.log)）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/h3world/bin/python submission/experiments/EXP-016_v3_real56_vae/run_vae.py --encode \
  > submission/experiments/EXP-016_v3_real56_vae/P1.log 2>&1
```

## 结果

| 场景 | 前12latent exact | max_abs / relative_RMS | 56帧roundtrip平均MAD | PSNR (dB) | 左源/右重建视频 |
| --- | --- | --- | ---: | ---: | --- |
| 43866101 | 是 | 0 / 0 | 5.165 | 29.736 | [MP4](artifacts/43866101/source_vs_reconstruction.mp4) |
| 7199292c | 是 | 0 / 0 | 5.889 | 29.469 | [MP4](artifacts/7199292c/source_vs_reconstruction.mp4) |
| 9dc2e588 | 是 | 0 / 0 | 5.792 | 29.251 | [MP4](artifacts/9dc2e588/source_vs_reconstruction.mp4) |
| b784d995 | 是 | 0 / 0 | 3.563 | 34.894 | [MP4](artifacts/b784d995/source_vs_reconstruction.mp4) |

MAD/PSNR由源RGB与VAE输出在再次MP4编码之前计算，均为全56帧均值；不作为生成视频质量指标。四条roundtrip右侧主体和场景在帧0、38、39、55均保持可辨，细节与草木纹理有VAE柔化；s3原录屏较暗且草遮挡人物，并非本轮编码造成的结构分解。边界预览：[s0](43866101_boundary_contact_sheet.jpg) · [s1](7199292c_boundary_contact_sheet.jpg) · [s2](9dc2e588_boundary_contact_sheet.jpg) · [s3](b784d995_boundary_contact_sheet.jpg)。完整影片还需Judge作正式视觉验收。

实际调用由[write-ahead账本](artifacts/budget.json)记录：**4 image encode + 8 video encode + 4 video decode、1次source VAE加载，0 text/DiT/训练**。GPU墙钟77.722秒（0.02159 GPU小时），`torch.cuda.max_memory_allocated`峰7.504GiB，低于600秒与44GiB上限。任务原始输出约41MiB；本轮14份日志/指标/MP4副本共24,643,157bytes，SHA与原始文件相同，见[归档清单](ARTIFACT_MANIFEST.json)。原始latent只保存在 `H3-World/outputs/EXP-016_v3_real56_vae/{scene}/I0.pt|full17.pt|prefix12.pt`，其SHA和shape见各场景[原始metrics](artifacts/43866101/metrics.json)及相应子目录，不复制大tensor到提交包。

Worker的[独立CPU审计](P1_CPU_AUDIT.json)重新加载12个保存tensor，四组前12逐值一致；8个输出MP4共448帧完整解码、24FPS、PTS递增，全部归档文件SHA通过。Judge另有[独立tensor/视频审计](judge/INDEPENDENT_RESULT_AUDIT.json)，其工程检查为PASS；视觉正式结论仍由Judge收口。

## 后续边界

本轮不触碰action文本、packed布局、DiT或模型训练。EXP-015的变长未来文本影响Global位置风险仍在；即使video VAE前缀完全一致，也不能把四段数据直接声明为完整V3训练ready。四段真实动作主要为A及混合键，没有真实D后果。下一步应先建立与action内容长度无关的packed位置协议，并对冻结V3做回归验证，另立任务后才考虑真实转移训练。
