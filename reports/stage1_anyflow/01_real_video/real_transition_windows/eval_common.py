"""Frozen E2 evaluation utilities. No optimizer, no shared hidden cache."""
import hashlib
from pathlib import Path
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def tensor_hash(x):
    x=x.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()


def move(x,device='cuda:0'):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    return x


def load_bank(model,path,*,precision,check_original=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['format']=='h3_online_lora_v1'
    assert state['metadata']['precision']==precision
    modules={getattr(m,'name',''):m for m in model.modules() if hasattr(m,'lora_A_weights')}
    values=[]
    for name,row in state['weights'].items():
        m=modules[name];assert len(m.lora_A_weights)==len(m.lora_B_weights)==1
        for key,weights in [('lora_A',m.lora_A_weights),('lora_B',m.lora_B_weights)]:
            value=row[key];assert torch.isfinite(value).all() and value.shape==weights[0].shape
            if check_original:assert torch.equal(value.float(),weights[0].detach().cpu().float())
            weights[0]=value.to('cuda:0');values.append(weights[0])
    assert len(values)==32
    return values,state['metadata']


def terminal(receipt):
    p=Path(f'/proc/{receipt["pid"]}/stat')
    if p.exists():
        fields=p.read_text().rsplit(')',1)[1].split()
        assert fields[19]!=receipt['start_ticks'] or fields[0] in ('Z','X'),'Original process is still active'
