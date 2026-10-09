"""Execute serial4, four-rank4, four-rank2+resume4 and compare real state."""
from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'outputs/2026-10-08-09/stage1_training_shift12/runtime'
COMMON=['--smoke','--objective','anyflow','--precision-profile','h3_fp32',
    '--adapter-scope','all_qkvo_ffn','--bank-rank','8','--bank-alpha','8',
    '--history-gradient-mode','full','--no-train-target-time','--logical-batch','4',
    '--lr','3e-5','--training-timestep-shift','12','--validation-timestep-shift','2.22',
    '--seed','13','--validation-seed','1001','--checkpoint-every','2']
PARALLEL=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(OUT/'run_trainer.py')]

def run(name,steps,parallel=True,resume=None):
    directory=OUT/name
    directory.mkdir(exist_ok=False)
    command=PARALLEL if parallel else [sys.executable,str(SOURCE/'code/causal/train_stage1_anyflow.py')]
    command=command+COMMON+['--steps',str(steps),'--out-dir',str(directory)]
    if resume:
        command += ['--resume-from',str(resume)]
    with (directory/'run.log').open('w') as log:
        subprocess.run(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True)
    s=json.loads((directory/'training.json').read_text())
    assert s['status']=='complete'
    if parallel:
        assert all(s['replica_audit']['equality'].values())
    print(name,'complete',flush=True)
    return directory

serial=run('cpu_serial4',4,False)
parallel=run('cpu_parallel4',4)
half=run('cpu_parallel2',2)
resumed=run('cpu_parallel_resume4',4,resume=half/'step_02')

import torch
torch.set_num_threads(2)

def load(directory,name):
    return torch.load(directory/name,map_location='cpu',weights_only=True)

def tensors(value,prefix=''):
    if torch.is_tensor(value):
        return {prefix:value}
    if isinstance(value,dict):
        return {k:v for key,item in value.items() if key not in ('metadata','config','updates')
                for k,v in tensors(item,prefix+'/'+str(key)).items()}
    if isinstance(value,(list,tuple)):
        return {k:v for i,item in enumerate(value) for k,v in tensors(item,prefix+'/'+str(i)).items()}
    return {}

result=dict(status='complete',at=datetime.now().astimezone().isoformat(),scope='CPU integrated trainer and actual Adam/RNG resume; no33B quality claim',comparisons={})
for name,directory,exact in [('parallel_vs_serial',serial,False),('parallel_resume_vs_full',resumed,True)]:
    entries={}
    for file in ('causal_adapter.pt','anyflow_adapter.pt','stage1_lora.pt','trainer_state.pt'):
        a=tensors(load(parallel,file));b=tensors(load(directory,file))
        assert a.keys()==b.keys()
        maximum=0.
        for key in a:
            if exact or not a[key].is_floating_point():
                assert torch.equal(a[key],b[key]),(name,file,key)
            else:
                torch.testing.assert_close(a[key],b[key],atol=6e-7,rtol=5e-4,msg=lambda m:f'{name}/{file}/{key}: {m}')
            if a[key].numel():
                maximum=max(maximum,float((a[key].double()-b[key].double()).abs().max()))
        entries[file]=dict(tensors=len(a),max_abs_difference=maximum,exact=exact)
    result['comparisons'][name]=entries
# Restore an OLD serial state into parallel and check four replica equality.
serial_resume=run('cpu_serial_to_parallel4',4,resume=serial/'step_02')
state=load(serial_resume,'trainer_state.pt')
reference=load(serial,'trainer_state.pt')
assert torch.equal(state['logical_rng_state'],reference['logical_rng_state'])
result['serial_to_parallel_resume']=dict(actual_old_serial_state_loaded=True,global_logical_RNG_matches=True,
    replica_audit=json.loads((serial_resume/'training.json').read_text())['replica_audit'])
(OUT/'cpu_training_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
print('Integrated CPU trainer/state checks complete',flush=True)
