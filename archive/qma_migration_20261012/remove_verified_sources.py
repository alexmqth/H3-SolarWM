from pathlib import Path
import os,json,shutil,subprocess,time
base=Path('/home/qma/work/.gwm_migration_20261012');old=Path('/home/lpeng/code/mq_PubDataset/GWM');new=Path('/home/qma/work/GWM');models=Path('/work/lpeng/qma/GWM_models')
assert json.loads((base/'runtime_verification.json').read_text())['status']=='PASS'
assert new.is_dir() and not new.is_symlink() and new.resolve()==new
assert new.stat().st_uid==os.getuid() and (new/'models').is_dir() and not (new/'models').is_symlink()
for name in ['project_checksum_differences.log','model_checksum_differences.log']:
 assert (base/name).stat().st_size==0
# Keep unpublished or concurrently edited source code from being removed.
for repo in ['submission','H3-World','H3-World/DiffSynth-Studio-h3-v2','SolarWM']:
 p=old/repo;a=['git','-c',f'safe.directory={p}','-C',str(p)]
 snap=json.loads((base/(repo.replace('/','_')+'_git_before.json')).read_text())
 status=subprocess.run(a+['status','--porcelain'],text=True,capture_output=True,check=True).stdout
 head=subprocess.run(a+['rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
 assert status==snap['status'] and head==snap['head'],('source changed',repo)
# Refuse to remove data while any unrelated executable is using the old project.
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(old) in args and not any(x in args for x in ['remove_verified_sources.py','codex','zsh -c','bash -c']):active.append({'pid':p.name,'args':args[:200]})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not active,active
assert str(old)=='/home/lpeng/code/mq_PubDataset/GWM' and str(models)=='/work/lpeng/qma/GWM_models'
record={'started_unix':time.time(),'verified_destination':str(new),'sources':[]}
for root in [old,models]:
 assert root.is_dir() and not root.is_symlink()
 before=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])
 print('Removing verified source contents:',root,'allocated bytes',before,flush=True)
 for child in list(root.iterdir()):
  if child.is_symlink() or not child.is_dir():child.unlink()
  else:shutil.rmtree(child)
 error=None
 try:root.rmdir()
 except PermissionError as e:error=str(e)
 assert not root.exists() or not list(root.iterdir())
 record['sources'].append({'path':str(root),'allocated_bytes_before':before,'exists_after':root.exists(),'contents_remaining':[],'rmdir_error':error,'allocated_bytes_after':root.stat().st_blocks*512 if root.exists() else 0})
 (base/'source_removal.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print('Source cleared:',root,'empty directory remains' if root.exists() else 'removed',flush=True)
record['finished_unix']=time.time();(base/'source_removal.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
