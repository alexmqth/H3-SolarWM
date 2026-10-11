from pathlib import Path
import os,json,shutil,hashlib
base=Path('/home/qma/work/.gwm_migration_20261012')
stage=Path('/home/qma/work/.GWM_migrating_20261012');final=Path('/home/qma/work/GWM');old=Path('/home/lpeng/code/mq_PubDataset/GWM')
# This script runs only after both complete content-comparison logs pass.
for name in ['project_checksum_differences.log','model_checksum_differences.log']:
 p=base/name
 assert p.is_file() and p.stat().st_size==0,(name,'content verification must finish first')
assert json.loads((base/'model_content_verification.json').read_text())['status']=='PASS'
assert json.loads((base/'project_content_verification.json').read_text())['status']=='PASS'
assert final.is_symlink() and final.resolve()==old
changed=json.loads((base/'symlink_relocations.json').read_text()) if (base/'symlink_relocations.json').exists() else []
for parent,dirs,files in os.walk(stage,followlinks=False):
 for n in dirs+files:
  p=Path(parent)/n
  if p.is_symlink():
   target=os.readlink(p)
   if target==str(old) or target.startswith(str(old)+'/') or target.startswith('/work/lpeng/qma/GWM_models'):
    if target.startswith('/work/lpeng/qma/GWM_models'):
     rel=Path('models')/Path(target).relative_to('/work/lpeng/qma/GWM_models')
    else:
     rel=Path(target).relative_to(old)
    new=os.path.relpath(stage/rel,p.parent)
    p.unlink();p.symlink_to(new);changed.append({'path':str(p.relative_to(stage)),'old_target':target,'new_target':new})
(base/'symlink_relocations.json').write_text(json.dumps(changed,indent=2)+'\n')
assert not any('/home/lpeng' in os.readlink(Path(par)/n) or '/work/lpeng' in os.readlink(Path(par)/n) for par,ds,fs in os.walk(stage,followlinks=False) for n in ds+fs if (Path(par)/n).is_symlink())
# Swap the existing alias for the verified real project; source is untouched here.
final.unlink();stage.rename(final)
assert final.is_dir() and not final.is_symlink() and final.stat().st_uid==os.getuid()
(base/'cutover.json').write_text(json.dumps({'old':str(old),'new':str(final),'new_is_real_directory':True,'owner_uid':os.getuid(),'relocated_symlinks':len(changed),'source_deleted':False},indent=2)+'\n')
print('Cutover complete:',final,'relative links relocated:',len(changed),flush=True)
