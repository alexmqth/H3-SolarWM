"""Four-process entrypoint for the isolated, unchanged-logical-batch trainer."""
from datetime import timedelta
from pathlib import Path
import os
import sys

OUT=Path(__file__).resolve().parent
RUNTIME=OUT/'training_runtime'
sys.path[:0]=[str(RUNTIME/'code'),str(RUNTIME/'code/abot'),str(RUNTIME/'DiffSynth-Studio-h3-v2')]
import torch
import torch.distributed as dist

rank=int(os.environ['LOCAL_RANK'])
smoke='--smoke' in sys.argv
if not smoke:
    torch.cuda.set_device(rank)
    if '--device' in sys.argv:
        sys.argv[sys.argv.index('--device')+1]=f'cuda:{rank}'
    else:
        sys.argv += ['--device',f'cuda:{rank}']
dist.init_process_group('gloo' if smoke else 'nccl',timeout=timedelta(minutes=15),
    **({} if smoke else dict(device_id=torch.device(f'cuda:{rank}'))))
try:
    if dist.get_world_size()!=4:
        raise ValueError('Exactly four ranks required; global logical batch remains4')
    from causal.train_stage1_anyflow import main
    main()
finally:
    dist.destroy_process_group()
