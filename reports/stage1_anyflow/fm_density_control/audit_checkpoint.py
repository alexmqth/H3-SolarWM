"""Read-only bounded-checkpoint audit; never modifies the running trainer."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

BASE=Path(__file__).resolve().parent
CONTROL=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
sys.path.insert(0,str(BASE/'runtime/code'))
from causal.training_state import load_training_state
from causal.anyflow import logical_time_pairs


def main(step):
    torch.set_num_threads(2)
    directory=BASE/f'train_48/step_{step:02d}'
    load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    state=load(directory/'trainer_state.pt')
    assert state['optimizer_step']==step and 0<step<=48
    config=dict(state['config'],steps=step+1)
    checked=load_training_state(directory,config)
    assert checked['optimizer_step']==step
    required=dict(objective='fm',training_timestep_shift=2.22,training_weight_shift=12.,
        validation_timestep_shift=2.22,flow_shift=2.22,logical_batch=4,lr=3e-5,seed=13,
        adapter_scope='all_qkvo_ffn',bank_rank=8,history_gradient_mode='full',anchor_mode='rgb',
        action_prefix_mode='causal',action_feedback=True,chunk_frames=5,history_chunks=5,
        causal_adapter=None,action_adapter=None)
    assert all(config[k]==v for k,v in required.items())
    assert config['steps']==step+1 and state['config']['steps']==48
    initial=BASE/'train_48/step_00'
    zero,visual=load(initial/'causal_adapter.pt'),load(directory/'causal_adapter.pt')
    for name in ('lora_A','lora_B'):
        assert all(torch.equal(a,b) for a,b in zip(zero[name],visual[name]))
    assert all(torch.count_nonzero(b)==0 for b in visual['lora_B'])
    first,bank=load(initial/'stage1_lora.pt'),load(directory/'stage1_lora.pt')
    assert first['targets']==bank['targets']
    changed=[]
    for name,weights in bank['weights'].items():
        pairs=[(a,b) for key in weights for a,b in zip(first['weights'][name][key],weights[key])]
        assert all(torch.isfinite(b).all() for a,b in pairs)
        if any(not torch.equal(a,b) for a,b in pairs):changed.append(name)
    assert len(changed)==len(bank['weights'])
    adam=state['optimizer']['state']
    assert adam and all(int(s['step'])==step for s in adam.values())
    assert all(torch.isfinite(s[k]).all() for s in adam.values() for k in ('exp_avg','exp_avg_sq'))
    expected=json.loads((BASE/'preflight.json').read_text())['samples']
    generator=torch.Generator().manual_seed(10013)
    for i,u in enumerate(state['updates']):
        assert u['step']==i+1 and u['chunk']==(i//16+i%16)%3
        pairs=logical_time_pairs(generator,shift=2.22)
        for j,s in enumerate(u['samples']):
            row=expected[i*4+j]
            assert (u['action'],u['chunk'],s['sigma'])==(row['clip'],row['chunk'],row['candidate_sigma'])
            assert s['sigma']==float(pairs.t[j])
            noise=torch.randn((1,24,5 if u['chunk']<2 else 2,30,52),generator=generator,dtype=torch.float32)
            assert hashlib.sha256(noise.numpy().tobytes()).hexdigest()==row['reconstructed_noise_sha256']
    assert torch.equal(generator.get_state(),state['logical_rng_state'])
    runtime=json.loads((BASE/'runtime_manifest.json').read_text())
    assert all(hashlib.sha256((BASE/'runtime'/p).read_bytes()).hexdigest()==h for p,h in runtime.items())
    receipt=dict(status='passed',at=datetime.now().astimezone().isoformat(),step=step,
        budget=48,weight_files_sha256=state['weight_sha256'],
        frozen_visual_still_zero=True,changed_bank_modules=len(changed),total_bank_modules=len(bank['weights']),
        finite_adam_tensors=True,all_adam_step_counters=step,
        curriculum_and_sigma_matches_preregistered_stream=True,
        logical_rng_matches_full_noise_replay=True,frozen_runtime_files=len(runtime),
        scope='Checkpoint/source/optimizer/input audit only; no video or action quality claim')
    (BASE/f'checkpoint{step:02d}_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--step',type=int,required=True)
    main(ap.parse_args().step)
