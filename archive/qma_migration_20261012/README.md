# qma项目迁移与旧实验空间清理（2026-10-12）

## 当前入口

项目实体目录：`/home/qma/work/GWM`，不再是指向lpeng的软链接。模型在项目内`models/`，文件与目录属于qma。使用：

```bash
cd /home/qma/work/GWM
source env.sh
python -c "import torch; print(torch.__version__)"
```

当前实验环境为`.venvs/h3world`，官方参考环境为`.venvs/solarwm-h3`；两者解释器均归qma。最新已验收实验/研究结论保持，无新GPU推理或训练。

## 清理范围与证据

- 44份已完成实验的大型KV缓存约439.035 GiB：已移除，保留端点、初始条件、源码、日志、指标和视频，必要时按原协议重新commit生成KV。保留EXP-002的`first12_A_cache.pt`和EXP-006的`cache_through12.pt`共约12.66 GiB，供后续基线/FM8对照复用。
- 240份早期日期目录中的trainer_state约23.390 GiB：移除optimizer等恢复状态；保存的adapter、评估视频和记录仍在。早期训练不能直接从这些已删optimizer状态恢复；新的EXP-007/008训练状态保留。
- 旧submission_acceptance重复环境约7.805 GiB与可重建编译/Python缓存已移除。
- MiniMax和SolarWM相同text encoder文件仅在完整SHA-256一致后用硬链接共用存储；两套模型路径均保留。基础模型应按只读资产使用，不就地修改硬链接权重。
- 旧根目录launcher和空package占位文件移到`legacy_root_files/`，防止误用过期GPU脚本。

精确逐文件清单见[cleanup_plan.json.gz](cleanup_plan.json.gz)、[清理摘要](cleanup_summary.json)、[模型去重](model_deduplication.json)。所有视频、报告、Git历史、冻结代码、关键latent与真实数据保留。清理不会把历史FAIL改成PASS；已删缓存不能当作当前仍在的实物证据。

## 旧路径如何使用

原始日志和冻结source manifest中的绝对路径保留原字节，映射关系见[path_mapping.json](path_mapping.json)。冻结runtime链中的绝对软链接已改为项目内相对链接；无需lpeng账号、解释器或模型目录。

历史路径读取工具：

```bash
python submission/scripts/project_paths.py /home/lpeng/code/mq_PubDataset/GWM/submission/experiments/EXP-017_v3_fixed_action_layout/fixed_layout.py
python submission/scripts/project_paths.py --json submission/experiments/EXP-016_v3_real56_vae/source_manifest.json
```

第二条打印迁移后的manifest视图，保留原记录。旧任务的hash门和已结束预算不会因搬家重新授权；若要重跑，建立新的任务并按新位置和已清理缓存情况准备输入。

## 校验与旧副本删除

项目文件的inode身份/SHA-256检查与模型的rsync逐文件XXH128内容checksum比对通过后，才把qma路径切成实体目录；[CPU/runtime检查](runtime_verification.json)验证最新EXP-017四场景CPU布局、冻结来源SHA、qma归属、无断链/无lpeng软链接依赖、报告链接及代表视频。模型前向和GPU调用均为0。

[旧源清理记录](source_removal.json)记录旧项目和外部模型目录删除情况。lpeng项目父目录对qma不可写：旧项目内部全部释放后，若顶层空目录仍在，需要lpeng或管理员执行`rmdir /home/lpeng/code/mq_PubDataset/GWM`；它已不含项目数据。

迁移不保证所有历史脚本可直接续跑：部分旧launcher硬编码了历史环境，部分旧审计要求已清理缓存，原路径记录也需要映射。当前H3/runtime/EXP-017输入与qma环境经过实际CPU验证；历史脚本作为证据保留。

## 最终结果

按本项目分配块统计，迁移前项目与外部模型合计约808.50 GiB，迁移后含模型约275.52 GiB，净减少约532.98 GiB（572.28 GB）。模型现约210.29 GiB，已含相同text encoder去重。旧`/work/lpeng/qma/GWM_models`已删除；旧home项目只剩4096字节空目录，内部无文件。统计和残余权限见[最终摘要](FINAL_SUMMARY.json)。
