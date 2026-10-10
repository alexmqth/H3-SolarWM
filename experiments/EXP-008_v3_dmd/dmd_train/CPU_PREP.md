# EXP-008/v2 DMD 延续训练 CPU 准备

2026-10-11 HKT。**CPU_PASS_NO_GPU_AUTHORIZATION，0 GPU 调用。** 本目录只准备从 v1 cycle01 checkpoint 恢复到累计 cycle8 的有限训练入口，尚未启动 v2 GPU。冻结任务见 [taskbook_v2.md](../taskbook_v2.md)。

入口：[run_v2.py](run_v2.py)；[config_v2.json](config_v2.json)；[source_manifest_v2.json](source_manifest_v2.json)，SHA `a7594031c0cae5a6e79653394b04bad8c01c88fdd07d922aa46c64c62416a0ac`。先前未加入完整恢复核查的预检 manifest 保存在 [source_manifest_v2_pre_review.json](source_manifest_v2_pre_review.json)，不作为运行来源。实际执行预检命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-008_v3_dmd/dmd_train/run_v2.py --preflight
```

预检检查29项来源 SHA、AF2 step32/AF3证据、pilot17forward/3backward/3update账本、四文件checkpoint摘要、pilot配套metadata和student optimizer step1/fake step2。用与保存参数形状一致的 CPU 参数重建两个 AdamW 并真正加载其 state；检查保存的训练噪声 RNG 恢复后重放一致。实际导入 `causal.dmd` 与冻结路径/SHA一致，33B模型没有加载。

GPU runner 的冻结行为：teacher C1只prefill一次；新增cycle2–8分别AD/AA交替，每cycle student/fake以当前权重重建C1、student保留8-map图、fake对detach端点做一次FM更新和KV重建、teacher/fake在同renoise端点评分、student做一次DMD反向与更新。每cycle预记14forward、2backward、2update的逐调用账本；全部完成上限为99forward、14backward、14update、0VAE。只在cycle4/8或预算提前收口的最后完整cycle保存配套权重/优化器/RNG。单cycle前检查剩余时间，原始输出不覆盖。

**独立 TRAIN marker 仍缺失。** GPU入口要求 `dmd_train/TRAIN_GPU_AUTHORIZATION.json` 同时绑定 `approved=true`、`af2_step=32`、`parent_cycle=1`、`last_cycle=8`、`physical_gpu_ids=[0,2,5]` 和以上 source manifest SHA。放行前仍需核对实时空闲GPU；09:00 HKT后项目最多占用3卡。本CPU核查不证明七个真实33B cycle会完成，更不证明视频质量改善。
