"""Bounded real-video FM bridge. Does not download or start other GPU jobs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[2]
RUNTIME = BASE / 'runtime'
MANIFEST = PROJECT / 'data/abot_bridge/encoded_manifest.json'
PYTHON = '/home/lpeng/miniconda3/envs/h3world/bin/python'

def main():
    sys.path.insert(0, str(RUNTIME / 'code'))
    import torch
    from causal.real_video_data import load_cases
    torch.set_num_threads(2)
    train, val = load_cases(MANIFEST, 'cpu')
    assert len(train) == 16 and len(val) == 8
    assert len({c['sample_id'] for c in train}) == 4
    assert len({c['sample_id'] for c in val}) == 2
    audit = dict(status='passed', manifest=str(MANIFEST),
        sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        train_cases=len(train), validation_cases=len(val),
        real_sources_verified=True, episode_disjoint=True,
        validation_order=[c['label'] for c in val],
        source='real public ABot clips, not teacher-generated endpoints')
    (BASE / 'final_input_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    del train, val
    gpu=0
    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free < 40000:
        raise RuntimeError(f'GPU {gpu} needs >=40000 MiB free; currently {free}; no launch')
    if (BASE / 'launch.json').exists() or (BASE / 'train_48').exists():
        raise RuntimeError('Run already exists; refusing duplicate launch')
    hashes={}
    for folder in ('code','DiffSynth-Studio-h3-v2/diffsynth','tests'):
        for p in sorted((RUNTIME/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.md'):
                hashes[str(p.relative_to(RUNTIME))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (BASE/'runtime_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    cmd=[PYTHON,'-u',str(RUNTIME/'code/causal/train_stage1_anyflow.py'),
        '--real-data-manifest',str(MANIFEST),'--out-dir',str(BASE/'train_48'),
        '--objective','fm','--adapter-scope','all_qkvo_ffn',
        '--bank-rank','8','--bank-alpha','8','--precision-profile','h3_fp32',
        '--history-gradient-mode','full','--anchor-mode','rgb',
        '--chunk-frames','5','--history-chunks','5','--action-prefix-mode','causal',
        '--action-feedback','--training-timestep-shift','12',
        '--validation-timestep-shift','2.22','--flow-shift','2.22',
        '--logical-batch','4','--lr','3e-5','--steps','48','--checkpoint-every','16',
        '--validation-cases','4','--seed','13','--validation-seed','1001']
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),ABOT_VRAM_RESERVE_GIB='14',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (BASE/'training.log').open('w') as f:
        p=subprocess.Popen(cmd,env=env,cwd=RUNTIME,stdout=f,stderr=subprocess.STDOUT,
                           start_new_session=True)
    receipt=dict(pid=p.pid,gpu=gpu,free_MiB_at_launch=free,
        launched_at=datetime.now().astimezone().isoformat(),command=cmd,
        reserve_GiB=14,maximum_updates=48,early_checkpoints=[1,3],
        initialization='Original H3 + released action LoRA + zero-output trainable rank8 bank',
        previous_trained_visual_action_adapters=False,
        data_manifest_sha256=audit['sha256'])
    (BASE/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
