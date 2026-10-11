from pathlib import Path
import gzip,json,shutil,os
root=Path('/home/qma/work/GWM');base=Path('/home/qma/work/.gwm_migration_20261012');arc=root/'submission/archive/qma_migration_20261012';arc.mkdir(parents=True,exist_ok=True)
assert json.loads((base/'runtime_verification.json').read_text())['status']=='PASS'
for name in ['cleanup_plan.json','output_inventory.json']:
 with (base/name).open('rb') as src,gzip.open(arc/(name+'.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
for p in base.iterdir():
 if p.is_file() and p.name not in ['cleanup_plan.json','output_inventory.json','project_paths.py','env.sh']:
  shutil.copy2(p,arc/p.name)
if (base/'cpu_validation').is_dir():shutil.copytree(base/'cpu_validation',arc/'cpu_validation',dirs_exist_ok=True)
shutil.copy2(base/'project_paths.py',root/'submission/scripts/project_paths.py');shutil.copy2(base/'env.sh',root/'env.sh');shutil.copy2(base/'env.sh',arc/'env.sh')
# Move obsolete top-level experiment launchers into the migration archive.
legacy=arc/'legacy_root_files';legacy.mkdir(exist_ok=True);moved=[]
for name in ['monitor_scheduled_sampling.sh','rerun_scheduled_sampling_lr3e4.sh','summarize_results.sh','test.sh','wait_and_benchmark.sh','package.json','package-lock.json']:
 p=root/name
 if p.exists():shutil.move(str(p),str(legacy/name));moved.append(name)
(arc/'root_cleanup.json').write_text(json.dumps({'archived_root_files':moved},indent=2)+'\n')
text='''# qma项目迁移与旧实验空间清理（2026-10-12）

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
python submission/scripts/project_paths.py /home/lpeng/code/mq_PubDataset/GWM/H3-World/examples/first_frame.png
python submission/scripts/project_paths.py --json submission/experiments/EXP-016_v3_real56_vae/source_manifest.json
```

第一条仅示范路径映射，具体文件请使用实际存在的路径。第二条打印迁移后的manifest视图，保留原记录。旧任务的hash门和已结束预算不会因搬家重新授权；若要重跑，建立新的任务并按新位置和已清理缓存情况准备输入。

## 校验与旧副本删除

项目文件的inode身份/SHA-256检查与模型的rsync逐文件XXH128内容checksum比对通过后，才把qma路径切成实体目录；[CPU/runtime检查](runtime_verification.json)验证最新EXP-017四场景CPU布局、冻结来源SHA、qma归属、无断链/无lpeng软链接依赖、报告链接及代表视频。模型前向和GPU调用均为0。

[旧源清理记录](source_removal.json)记录旧项目和外部模型目录删除情况。lpeng项目父目录对qma不可写：旧项目内部全部释放后，若顶层空目录仍在，需要lpeng或管理员执行`rmdir /home/lpeng/code/mq_PubDataset/GWM`；它已不含项目数据。

迁移不保证所有历史脚本可直接续跑：部分旧launcher硬编码了历史环境，部分旧审计要求已清理缓存，原路径记录也需要映射。当前H3/runtime/EXP-017输入与qma环境经过实际CPU验证；历史脚本作为证据保留。
'''
# Use an actual known file in the lookup example.
text=text.replace('H3-World/examples/first_frame.png','submission/experiments/EXP-017_v3_fixed_action_layout/fixed_layout.py').replace('第一条仅示范路径映射，具体文件请使用实际存在的路径。第二条','第二条')
(arc/'README.md').write_text(text)
for name in ['README.md','progress.md','next_plan.md','guideline.md','archive.md']:
 p=root/name;t=p.read_text()
 if name=='README.md':
  t=t.replace('\n\n','\n\n**项目已于2026-10-12迁移为`/home/qma/work/GWM`实体目录，所有模型位于`models/`。** 使用`source env.sh`进入qma运行环境；[迁移与清理记录](submission/archive/qma_migration_20261012/README.md)。\n\n',1)
 else:
  t+='\n\n## 2026-10-12 qma迁移与空间清理\n\n用户明确授权迁移项目并清理旧实验。当前实体目录`/home/qma/work/GWM`，模型在`models/`，qma拥有全部保留文件。已清理可重建KV、早期optimizer状态及重复环境，保留正式证据/权重和后续输入；已删除文件不再声称实物仍可检查。详见[迁移记录](submission/archive/qma_migration_20261012/README.md)。本次无GPU/训练，新研究任务仍需独立预算与Judge放行。\n'
 p.write_text(t)
p=root/'submission/README.md';t=p.read_text();t=t.replace('\n\n','\n\n**服务器实体目录已迁移到`/home/qma/work/GWM`，模型位于其`models/`，统一使用qma账号。** [迁移、旧实验清理和运行入口](archive/qma_migration_20261012/README.md)。\n\n',1);p.write_text(t)
p=root/'submission/REPRODUCE.md';t=p.read_text();t=t.replace('\n\n','\n\n2026-10-12迁移更新：从`/home/qma/work/GWM`实体目录运行，先`source env.sh`。历史绝对路径需按[迁移记录](archive/qma_migration_20261012/README.md)映射，部分大型KV/旧optimizer缓存已清理，重跑相关步骤需重新生成；冻结原始记录保持不变。\n\n',1);p.write_text(t)
snap=arc/'management_snapshot';snap.mkdir(exist_ok=True)
for name in ['README.md','progress.md','next_plan.md','guideline.md','archive.md']:shutil.copy2(root/name,snap/name)
print('Migration docs and portable path resolver installed:',arc)
