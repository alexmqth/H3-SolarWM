from pathlib import Path
import json,hashlib,subprocess
s=Path(__file__).resolve().parents[2]; a=s/'archive/v3_baseline_restore_20261010'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((a/'migration.json').read_text()); results=[]
for row in m['files']:
 if row['new'].endswith('.mp4'): row['new']=row['new'].replace('V2c','V3_Baseline')
 p=s/row['new']; current=sha(p); same=current==row['sha256']; result=dict(row,current_sha256=current,unchanged=same)
 if not same:
  if p.suffix=='.mp4':
   assert '/8step_continuation/videos/' in row['new'], row
   name=p.name.replace('V3_Baseline','V3')
   source=s/'experiments/EXP-004_v3_8step/artifacts/videos'/name
   assert sha(source)==current
   oldcopy=a/'previous_display_videos'/row['old']
   candidates=list((a/'previous_display_videos').rglob(Path(row['old']).name))
   assert len(candidates)==1 and sha(candidates[0])==row['sha256'],candidates
   result['replacement_source']=str(source.relative_to(s)); result['old_archived']=str(candidates[0].relative_to(s))
  else: assert p.name in ('README.md','metrics.json','release_manifest.json'),row
 results.append(result)
(a/'migration.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
manifest=json.loads((s/'mainline/v3/v3_baseline/release_manifest.json').read_text()); checks={}
for n,h in manifest['frozen_code'].items():checks['code/'+n]=sha(s/'experiments/EXP-003_native_cached_124'/n)==h
checks['config']=sha(s/'experiments/EXP-003_native_cached_124/config.json')==manifest['config_sha256']
for n,h in manifest['videos'].items():checks['videos/'+n]=sha(s/'report/v3/v3_baseline/videos'/n)==h
assert all(checks.values())
# Historical source, raw reports, metrics and experiment artifacts are unchanged,
# except current-facing README and taxonomy fields explicitly allowed above.
for exp in ('EXP-002_native_cached','EXP-003_native_cached_124'):
 for f in subprocess.check_output(['git','-c',f'safe.directory={s.resolve()}','-C',str(s),'ls-tree','-r','--name-only',m['base_commit'],f'experiments/{exp}'],text=True).splitlines():
  if Path(f).name in ('README.md','metrics.json'):continue
  old=subprocess.check_output(['git','-c',f'safe.directory={s.resolve()}','-C',str(s),'show',f"{m['base_commit']}:{f}"])
  assert hashlib.sha256(old).hexdigest()==sha(s/f),f
  checks[f]=True
out={'base_commit':m['base_commit'],'mapped_files':len(results),'unchanged_files':sum(x['unchanged'] for x in results),'files':results,'frozen_checks':checks,'all_passed':True}
(a/'validation.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'mapped':len(results),'unchanged':sum(x['unchanged'] for x in results),'frozen_checks':len(checks),'pass':True}))
