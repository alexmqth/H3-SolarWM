"""Archive completed stage logs/videos/metrics, leaving large tensors external."""
from pathlib import Path
import json,hashlib,shutil
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=EXP.parents[2]
SRC=ROOT/'H3-World/outputs/EXP-005_v3_sliding_window'
DEST=EXP/'artifacts/stage2';DEST.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
rows=[]
for stage in ['G0','G1','L1']:
 src=SRC/stage
 if not (src/'budget.json').exists():continue
 budget=json.loads((src/'budget.json').read_text())
 if budget['status']=='running':raise RuntimeError(f'{stage} still running')
 for p in sorted(src.rglob('*')):
  if not p.is_file() or p.suffix not in ['.json','.log','.mp4','.jpg']:continue
  out=DEST/stage/p.relative_to(src);out.parent.mkdir(parents=True,exist_ok=True)
  digest=sha(p)
  if out.exists():assert sha(out)==digest, f'archive would overwrite evidence: {out}'
  else:shutil.copy2(p,out)
  rows.append({'source':str(p),'archived':str(out.relative_to(EXP)),'sha256':digest,'bytes':p.stat().st_size})
(EXP/'artifact_manifest_stage2.json').write_text(json.dumps({'files':rows,'external_large_artifacts':str(SRC),'excluded':['.pt','.npy'],'no_inference':True},indent=2)+'\n')
print('archived',len(rows),'small evidence files')
