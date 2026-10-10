"""CPU/file-only validation of current navigation and frozen source hashes."""
from pathlib import Path
import hashlib,json,re
from urllib.parse import unquote
HERE=Path(__file__).resolve().parent
S=HERE.parents[2]
ROOT=S.parent
paths=set(ROOT/p for p in ['README.md','guideline.md','progress.md','next_plan.md','report.md'])
paths.update(S/p for p in ['README.md','REPORT.md','INTERVIEW_ANSWER.md','REORGANIZATION_SUMMARY.md'])
for folder in ['report','mainline']:
 paths.update((S/folder).rglob('*.md')); paths.update((S/folder).rglob('*.html'))
for name in ['EXP-002_native_cached','EXP-003_native_cached_124','EXP-004_v3_8step']:
 paths.add(S/'experiments'/name/'README.md')
E=HERE.parent
paths.update(E.glob('*.md')); paths.add(HERE/'FINAL_REVIEW.md')
paths={p for p in paths if not p.name.startswith(('taskbook_','worker_','guideline_snapshot'))}
errors=[];count=0
for p in sorted(paths):
 text=p.read_text()
 targets=re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)',text)+re.findall(r'(?:href|src)=[\"\']([^\"\']+)[\"\']',text)
 for target in targets:
  target=target.strip().split(' "')[0].strip('<>')
  if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
  target=unquote(target.split('#')[0].split('?')[0])
  if not target:continue
  count+=1
  if not (p.parent/target).exists():errors.append({'source':str(p.relative_to(ROOT)),'target':target})
hashes=[]
for path,sha in re.findall(r'\|[^\n]*?\| `([^`]+)` \| `([0-9a-f]{64})` \|',(E/'MANIFEST.md').read_text()):
 p=ROOT/path
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 hashes.append({'path':path,'sha256':actual,'pass':sha==actual})
 if sha!=actual:errors.append({'source':'MANIFEST.md','hash_mismatch':path})
root_md=sorted(p.name for p in ROOT.glob('*.md'))
assert root_md==['README.md','archive.md','guideline.md','next_plan.md','progress.md','report.md'],root_md
result={'local_links_checked':count,'source_hashes_checked':len(hashes),'root_markdown_files':root_md,'errors':errors,'hashes':hashes}
(HERE/'delivery_checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='hashes'},ensure_ascii=False))
raise SystemExit(bool(errors))
