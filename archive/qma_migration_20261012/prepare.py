from pathlib import Path
import json,os,subprocess
base=Path('/home/qma/work/.gwm_migration_20261012');src=Path('/home/lpeng/code/mq_PubDataset/GWM');dst=Path('/home/qma/work/.GWM_migrating_20261012')
assert not dst.exists(); dst.mkdir(mode=0o755)
rows=json.loads((base/'output_inventory.json').read_text())
keep={
'H3-World/outputs/EXP-002_native_cached/first12_A_cache.pt',
'H3-World/outputs/EXP-006_v3_fm8_full/cache_through12.pt',
}
remove=[];by={}
for row in rows:
 p=Path(row['path']);reason=None
 if '/submission_acceptance/venv/' in '/'+row['path']:reason='obsolete_acceptance_virtualenv'
 elif p.name=='trainer_state.pt' and '/2026-' in row['path']:reason='legacy_optimizer_resume_state'
 elif p.suffix=='.pt' and 'cache' in p.name and row['size']>2**30 and row['path'] not in keep:reason='rebuildable_completed_experiment_KV_cache'
 if reason:
  remove.append(dict(row,reason=reason));by[reason]=by.get(reason,0)+row['allocated']
ex=['/models','__pycache__/','.pytest_cache/','/H3-World/.cache/torchinductor/','/H3-World/.cache/triton/']
ex += ['/'+x['path'] for x in remove if x['reason']!='obsolete_acceptance_virtualenv']
ex += ['/H3-World/outputs/2026-10-09-16/submission_acceptance/venv/']
(base/'project_excludes.txt').write_text('\n'.join(ex)+'\n')
(base/'cleanup_plan.json').write_text(json.dumps({'source':str(src),'destination':str(dst),'kept_reusable_KV':sorted(keep),'removed_files':remove,'allocated_bytes_by_reason':by,'preserved':'All source/config/log/metrics/video/image/latent/endpoints, model adapters, latest EXP optimizer states, true data and both primary environments; all Git histories. Only listed KV, legacy optimizer states, redundant acceptance environment and compiled caches omitted.','model_source':'/work/lpeng/qma/GWM_models','model_destination':'models/','source_root_parent_writable':os.access(src.parent,os.W_OK)},ensure_ascii=False,indent=2)+'\n')
for repo in ['submission','H3-World','H3-World/DiffSynth-Studio-h3-v2','SolarWM']:
 p=src/repo
 a=['git','-c',f'safe.directory={p}','-C',str(p)]
 record={'status':subprocess.run(a+['status','--porcelain'],text=True,capture_output=True).stdout,'head':subprocess.run(a+['rev-parse','HEAD'],text=True,capture_output=True).stdout.strip()}
 (base/(repo.replace('/','_')+'_git_before.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'target':str(dst),'planned_removed_GiB':{k:round(v/2**30,2) for k,v in by.items()},'preserved_KV_GiB':12.66},indent=2))
