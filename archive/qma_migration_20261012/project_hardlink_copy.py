from pathlib import Path
import json,os,shutil,stat,hashlib,time
base=Path('/home/qma/work/.gwm_migration_20261012');src=Path('/home/lpeng/code/mq_PubDataset/GWM');dst=Path('/home/qma/work/.GWM_migrating_20261012')
assert src.stat().st_dev==dst.stat().st_dev
plan=json.loads((base/'cleanup_plan.json').read_text());omit={x['path'] for x in plan['removed_files']}
prefixes=['models','H3-World/outputs/2026-10-09-16/submission_acceptance/venv','H3-World/.cache/torchinductor','H3-World/.cache/triton']
def excluded(rel):return any(x in Path(rel).parts for x in ['__pycache__','.pytest_cache']) or rel in omit or any(rel==x or rel.startswith(x+'/') for x in prefixes)
counts={'hardlinked_files':0,'hardlinked_bytes':0,'copied_files':0,'copied_bytes':0,'symlinks':0,'directories':0};copied=[];started=time.time()
for parent,dirs,files in os.walk(src,followlinks=False):
 rel=Path(parent).relative_to(src);target=dst/rel;target.mkdir(parents=True,exist_ok=True)
 dirs[:]=[d for d in dirs if not excluded(str(rel/d))]
 for n in dirs+files:
  p=Path(parent)/n;relp=p.relative_to(src);r=str(relp)
  if excluded(r):continue
  q=dst/relp;s=p.lstat()
  if p.is_symlink():
   if q.is_symlink():q.unlink()
   elif q.exists():raise RuntimeError(('unexpected existing path',str(q)))
   q.symlink_to(os.readlink(p));counts['symlinks']+=1
  elif p.is_dir():
   q.mkdir(exist_ok=True);counts['directories']+=1
  elif p.is_file():
   if s.st_uid==os.getuid():
    if q.exists() and os.path.samefile(p,q):continue
    tmp=q.with_name(q.name+'.gwm_link_tmp');os.link(p,tmp);os.replace(tmp,q)
    assert os.path.samefile(p,q)
    counts['hardlinked_files']+=1;counts['hardlinked_bytes']+=s.st_size
   else:
    shutil.copy2(p,q);os.chmod(q,stat.S_IMODE(s.st_mode)|stat.S_IWUSR)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(q.read_bytes()).hexdigest()==h
    counts['copied_files']+=1;counts['copied_bytes']+=s.st_size;copied.append({'path':r,'sha256':h,'bytes':s.st_size})
 if counts['hardlinked_files'] and counts['hardlinked_files']%10000<10:print(counts,flush=True)
# Verify every retained path. A matching inode is stronger than comparing hashes.
verified=0;errors=[]
for parent,dirs,files in os.walk(src,followlinks=False):
 rel=Path(parent).relative_to(src);dirs[:]=[d for d in dirs if not excluded(str(rel/d))]
 for n in dirs+files:
  p=Path(parent)/n;r=str(p.relative_to(src));q=dst/r
  if excluded(r):continue
  if p.is_symlink():assert q.is_symlink() and os.readlink(p)==os.readlink(q),r
  elif p.is_file():
   assert q.is_file() and p.stat().st_size==q.stat().st_size,r
   if p.stat().st_uid==os.getuid():assert os.path.samefile(p,q),r
   else:assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(q.read_bytes()).digest(),r
  else:assert q.is_dir(),r
  verified+=1
result={'status':'PASS','method':'same-filesystem inode identity for qma-owned files; SHA-256 verified copies for lpeng-owned files; exact symlink target checks','counts':counts,'copied_foreign_owned_files':copied,'verified_retained_paths':verified,'elapsed_seconds':time.time()-started}
(base/'project_content_verification.json').write_text(json.dumps(result,indent=2)+'\n')
(base/'project_checksum_differences.log').write_text('')
print(json.dumps({k:v for k,v in result.items() if k!='copied_foreign_owned_files'},indent=2),flush=True)
