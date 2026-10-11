from pathlib import Path
import hashlib,json,os
base=Path('/home/qma/work/.gwm_migration_20261012');root=Path('/home/qma/work/.GWM_migrating_20261012/models')
a=root/'MiniMax-H3/FL2VA/text_encoder';b=root/'SolarWM/SolarWM-h3-33B-base/text_encoder'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
rows=[]
for p in sorted(b.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 q=a/p.relative_to(b)
 if not q.is_file() or p.stat().st_size!=q.stat().st_size:continue
 h=sha(p)
 if sha(q)!=h:continue
 size=p.stat().st_size;allocated=p.stat().st_blocks*512
 temporary=p.with_name(p.name+'.dedup_tmp')
 os.link(q,temporary);os.replace(temporary,p)
 rows.append({'canonical':str(q.relative_to(root)),'duplicate':str(p.relative_to(root)),'sha256':h,'bytes':size,'allocated_bytes_saved':allocated})
 print('deduplicated',p.name,round(size/2**30,3),'GiB',flush=True)
(base/'model_deduplication.json').write_text(json.dumps({'method':'hardlink only after matching full SHA-256 and size; both model paths preserved','files':rows,'allocated_bytes_saved':sum(x['allocated_bytes_saved'] for x in rows)},indent=2)+'\n')
