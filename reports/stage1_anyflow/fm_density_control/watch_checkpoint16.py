from pathlib import Path
import json,time,subprocess,hashlib,datetime,os
b=Path(__file__).resolve().parent
launch=json.loads((b/'launch.json').read_text())
expected=json.loads((b/'checkpoint16_watch_launch.json').read_text())['audit_source_sha256']
receipt=b/'checkpoint16_watch.json'
def save(status,**extra):
 data=dict(status=status,at=datetime.datetime.now().astimezone().isoformat(),training_pid=launch['pid'],**extra)
 temp=receipt.with_suffix('.tmp.json');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(receipt)
save('waiting_for_checkpoint')
try:
 while not (b/'train_48/step_16/trainer_state.pt').exists():
  p=Path(f"/proc/{launch['pid']}/stat");a=p.read_text().split(') ')[-1].split() if p.exists() else None
  if not a or a[0]=='Z' or a[19]!=launch['start_ticks']:raise RuntimeError('Original training process ended before checkpoint16; no retry')
  time.sleep(20)
 assert hashlib.sha256((b/'audit_checkpoint.py').read_bytes()).hexdigest()==expected
 save('auditing')
 with (b/'checkpoint16_audit.log').open('x') as f:
  subprocess.run(['/home/lpeng/miniconda3/envs/h3world/bin/python',str(b/'audit_checkpoint.py'),'--step','16'],stdout=f,stderr=subprocess.STDOUT,check=True)
 save('complete',audit='checkpoint16_audit.json')
except BaseException as e:
 save('failed',error=repr(e));raise
