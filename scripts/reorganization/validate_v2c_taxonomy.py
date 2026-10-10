"""Validate the taxonomy move, current links and unchanged frozen evidence on CPU."""
from pathlib import Path
import hashlib,json,re
S=Path(__file__).resolve().parents[2];R=S.parent;A=S/'archive/taxonomy_v2c_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((A/'before_moved_files.json').read_text());relabels=json.loads((A/'display_video_relabels.json').read_text());display={x['target'] for x in relabels};same=[];changed=[];protected=[]
for row in rows:
 p=S/row['new'];assert p.is_file(),row
 if sha(p)==row['sha256']:same.append(row['new'])
 else:
  assert p.suffix in {'.md','.json'} or row['new'] in display,('unexpected byte change',row)
  changed.append(row['new'])
 if (p.suffix in {'.py','.log','.mp4','.pt','.patch'} and row['new'] not in display) or any(x in p.name for x in ['worker_report','worker_metrics','taskbook','guideline_snapshot']):
  assert sha(p)==row['sha256'],('frozen evidence changed',row['new'])
  protected.append(row['new'])
for row in relabels:
 assert sha(S/row['source'])==row['source_sha256'] and sha(S/row['target'])==row['sha256']
files=list((S/'report').rglob('*.md'))+list((S/'mainline').rglob('*.md'))+[S/'report/index.html']+[S/x for x in ['README.md','REPORT.md','INTERVIEW_ANSWER.md','REPRODUCE.md','REORGANIZATION_SUMMARY.md']]+[R/x for x in ['README.md','guideline.md','progress.md','next_plan.md','report.md']]+[S/'experiments'/x/'README.md' for x in ['EXP-002_native_cached','EXP-003_native_cached_124','EXP-004_v2c_8step']]
n=0
for f in files:
 for link in re.findall(r'\]\(([^)]+)\)',f.read_text())+re.findall(r'(?:href|src|value)="([^"]+)"',f.read_text()):
  if '://' in link or link.startswith(('#','mailto:','data:')):continue
  n+=1;assert (f.parent/link.split('#')[0]).exists(),('broken current link',str(f),link)
for f in (S/'mainline').rglob('*.json'):json.loads(f.read_text())
for area in ['report','mainline']:
 assert sorted(x.name for x in (S/area/'v2').iterdir() if x.is_dir())==['v2a_rgb_anchor','v2b_same_sigma_local_bidir','v2c_strict_causal_kv']
 assert not any(p.name.startswith(('V2','V3')) for p in (S/area).iterdir())
assert (S/'experiments/EXP-004_v3_8step').is_symlink()
assert (S/'experiments/EXP-004_v3_8step').resolve()==(S/'experiments/EXP-004_v2c_8step').resolve()
release=json.loads((S/'mainline/v2/v2c_strict_causal_kv/release_manifest.json').read_text())
for filename,digest in release['videos'].items():assert sha(S/'report/v2/v2c_strict_causal_kv/videos'/filename)==digest
assert sorted(p.name for p in R.glob('*.md'))==['README.md','archive.md','guideline.md','next_plan.md','progress.md','report.md']
result=dict(moved_files=len(rows),byte_identical=len(same),frozen_evidence_identical=len(protected),allowed_current_document_or_display_changes=changed,checked_current_links=n,broken_current_links=0,display_videos_relabelled=2,formal_124frame_video_hashes_match=True,exp004_compatibility_alias=True,new_model_forwards=0,new_training_updates=0,root_markdown_count=6)
(A/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
