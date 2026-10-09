"""Reconstruct the exact logical CPU sampler and audit completed fork updates.

Hashes below describe reconstructed FP32 CPU noise. The live trainer did not
log actual GPU noise tensors, so this does not claim a direct GPU tensor audit.
"""
from pathlib import Path
import hashlib
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128/train_128/step_128'
sys.path[:0]=[str(OUT/'runtime/code'),str(OUT/'runtime/DiffSynth-Studio-h3-v2')]
import torch
from causal.anyflow import logical_time_pairs


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


torch.set_num_threads(4)
state=torch.load(SOURCE/'trainer_state.pt',map_location='cpu',weights_only=True)
generator=torch.Generator().set_state(state['logical_rng_state'])
shapes={a:torch.load(ROOT/f'outputs/2026-10-02-03/action_{a}_teacher_39/baseline_latents.pt',
                    map_location='cpu',weights_only=True).shape for a in ('A','D')}
expected=[]
states={128:generator.get_state().clone()}
for step in range(128,136):
    action=('A','D')[step%2];chunk=(step//2)%3
    shape=list(shapes[action]);shape[2]=min(5,shape[2]-chunk*5)
    pairs=logical_time_pairs(generator,shift=12.,batch_size=4)
    samples=[]
    for i in range(4):
        noise=torch.randn(shape,generator=generator,dtype=torch.float32)
        samples.append(dict(sigma=float(pairs.t[i]),target_sigma=float(pairs.r[i]),
            sample_type=('diffusion','endpoint','flow_map')[int(pairs.sample_type[i])],
            reconstructed_cpu_noise_sha256=sha(noise),shape=shape))
    expected.append(dict(step=step+1,action=action,chunk=chunk,samples=samples))
    states[step+1]=generator.get_state().clone()

result=dict(scope='Logical sampler audit, not directly logged GPU-noise hashes',
    source=str(SOURCE),expected=expected,variants={},complete_pair=False)
for variant in ('control','auxiliary'):
    directory=OUT/variant/'train_136';file=directory/'training.json'
    if not file.exists():
        result['variants'][variant]=dict(status='not_started');continue
    training=json.loads(file.read_text())
    updates=[x for x in training['updates'] if x['step']>128]
    verified=[]
    for update in updates:
        row=expected[update['step']-129]
        assert (update['step'],update['action'],update['chunk'])==(row['step'],row['action'],row['chunk'])
        assert len(update['samples'])==4
        for sample,reference in zip(update['samples'],row['samples']):
            assert all(sample[key]==reference[key] for key in ('sigma','target_sigma','sample_type'))
        verified.append(update['step'])
    checkpoints={}
    for step in (128,132,136):
        checkpoint=directory/f'step_{step}/trainer_state.pt'
        if not checkpoint.exists():continue
        # A trainer may be between atomic state-file writes and ancillary files;
        # trainer_state itself is written through an atomic rename.
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['optimizer_step']==step
        assert torch.equal(saved['logical_rng_state'],states[step])
        checkpoints[str(step)]=dict(logical_rng_matches_reconstruction=True,
                                   sha256=sha(saved['logical_rng_state']))
    result['variants'][variant]=dict(status=training['status'],verified_steps=verified,
                                     checkpoint_rng=checkpoints)
result['complete_pair']=all(v.get('verified_steps')==list(range(129,137)) for v in result['variants'].values())
(OUT/'sample_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='expected'},indent=2))
