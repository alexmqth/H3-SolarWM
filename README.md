# H3-SolarWM：因果世界模型研究

**核心对比：[Original H3-World vs V3因果版本：视频、推理耗时、峰值显存、质量与连续性](report/original_vs_causal/README.md)。** 包含正式Baseline和SW-G + FM-8NFE候选，统计范围与缺项单独说明。

**命名规范（2026-10-12）：** `AF-train-32updates`是训练累计更新状态；`AF-8NFE` / `FM-8NFE`是每块推理求值配置；EXP-007 / AF2是训练阶段编号。[完整映射与计数说明](report/GLOSSARY.md)。

**服务器实体目录已迁移到`/home/qma/work/GWM`，模型位于其`models/`，统一使用qma账号。** [迁移、旧实验清理和运行入口](archive/qma_migration_20261012/README.md)。

**已验收正式参考：[V3 Baseline](mainline/v3/v3_baseline/README.md)**，严格因果、持久KV、Global RoPE，30步124帧有限可行。画质与连续性仍PARTIAL。

## 按研究方向看报告和视频

- [V2：RGB Anchor / Same-σ](report/v2/README.md)：视觉与动作修复路线。
- [V3 Baseline](report/v3/v3_baseline/README.md)：正式124帧参考。
- [V3 Sliding Window](report/v3/sliding_windows/README.md)：Global / Local位置编码；Global158帧有限可行，当前Local无联合收益。
- [V3 FM → AnyFlow](report/v3/fm2anyflow/README.md)：普通FM-30NFE/8/4、AnyFlow8/4、场景迁移和训练数据准备。FM-8NFE是每块8步普通采样。
- [V4 SGF / DMD](report/v4/README.md)：训练工程通过，当前DMD配置生成失败；早期旧协议尝试单列。

[完整报告导航](report/README.md) · [按路线播放视频](report/index.html) · [缩写与数字含义](report/GLOSSARY.md) · [主线定义](mainline/README.md) · [复现](REPRODUCE.md)

## 当前状态

截至2026-10-11 09:00获批研究已收口。EXP-015/016完成真实数据和VAE验证；EXP-017固定动作位置布局仅通过CPU验收，实际模型/生成尚未测试。[下一阶段草案](report/02_next_steps/README.md)建议先做实际模型回归与真实混合动作的FM-30NFE/FM-8NFE局部续写，尚未启动。

`report/`保存易读说明和代表性视频，`experiments/`保存任务原始证据。V4是研究分类，当前无已验收V4生成模型；原EXP目录、checkpoint与冻结结果保留。根目录progress.md和next_plan.md分别管理正式状态与执行授权。
