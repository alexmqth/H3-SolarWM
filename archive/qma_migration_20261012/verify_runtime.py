from pathlib import Path
import os,json,sys,hashlib,importlib.util
root=Path('/home/qma/work/GWM');base=Path('/home/qma/work/.gwm_migration_20261012');out=base/'cpu_validation';out.mkdir(exist_ok=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
assert root.resolve()==root and not root.is_symlink()
assert Path(sys.prefix)==root/'.venvs/h3world',sys.prefix
import torch,av,transformers
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
# Actual latest CPU candidate, frozen source unchanged; redirect only its new result file.
p=root/'submission/experiments/EXP-017_v3_fixed_action_layout/fixed_layout.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='fc4a47e8070aa0e9214c486396367227ca413e97e9056357dd318efc22d0d7d2'
spec=importlib.util.spec_from_file_location('migrated_fixed_layout',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.HERE=out;m.main()
# Verify the frozen native runtime, input fixtures and released action adapter against its evidence.
manifest=json.loads((root/'submission/experiments/EXP-006_v3_fm8_full/source_manifest.json').read_text());verified=[]
for name,row in manifest['sources'].items():
 p=root/row['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
 assert h.hexdigest()==row['sha256'],(name,str(p))
 verified.append(name)
links=[];foreign=[]
for parent,dirs,files in os.walk(root,followlinks=False):
 for name in dirs+files:
  p=Path(parent)/name;s=p.lstat()
  if s.st_uid!=os.getuid():foreign.append(str(p.relative_to(root)))
  if p.is_symlink():
   resolved=str(p.resolve())
   if not p.exists() or resolved.startswith(('/home/lpeng','/work/lpeng')):links.append({'path':str(p),'target':os.readlink(p),'resolved':resolved})
assert not foreign,foreign[:10]
assert not links,links
# Check report links and byte-preserved representative videos without model execution.
import re
broken=[];count=0
for d in [root/'submission/report',root/'submission/mainline']:
 for p in d.rglob('*'):
  if p.suffix not in ['.md','.html']:continue
  for a,b in re.findall(r'\]\(([^\s)]+)\)|(?:href|src|value|data-report)="([^"]+)"',p.read_text()):
   u=a or b
   if re.match(r'^(?:[a-z]+:|#)',u):continue
   target=u.split('#')[0]
   if not target:continue
   count+=1
   if not (p.parent/target).exists():broken.append({'file':str(p),'link':u})
assert not broken,broken
videos=[]
for rel in ['submission/report/v3/v3_baseline/videos/V3_Baseline_AA_124.mp4','submission/report/v3/fm2anyflow/fm8/videos/A_SWG30_vs_FM8_SWG8_158.mp4','submission/report/v4/dmd/videos/AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4']:
 with av.open(str(root/rel)) as c:
  s=c.streams.video[0];s.thread_count=1;frame=next(c.decode(s));videos.append({'path':rel,'frames':s.frames,'decoded_first_frame':True})
record={'status':'PASS','python':sys.executable,'prefix':sys.prefix,'torch':torch.__version__,'transformers':transformers.__version__,'latest_CPU_candidate':'four scenes passed; actual model generation not tested','frozen_sources_hash_verified':verified,'broken_symlinks':links,'foreign_owned_paths':foreign,'report_links_checked':count,'broken_report_links':broken,'videos':videos,'model_forward_calls':0,'gpu_calls':0}
(base/'runtime_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
