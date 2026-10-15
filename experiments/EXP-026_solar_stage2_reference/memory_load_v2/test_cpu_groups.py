"""Exercise official physical-node HYBRID_SHARD group creation using Gloo only."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from solarwm.backends.minimax_h3.fsdp import frozen_base_groups

assert int(os.environ['WORLD_SIZE']) == 8
assert int(os.environ['LOCAL_WORLD_SIZE']) == 8
dist.init_process_group('gloo', init_method='env://')
try:
    shard, replica = frozen_base_groups(8)
    result = {
        'rank': dist.get_rank(),
        'shard_group_size': dist.get_world_size(shard),
        'replica_group_size': dist.get_world_size(replica),
        'cuda_initialized': torch.cuda.is_initialized(),
    }
    gathered = [None] * 8
    dist.all_gather_object(gathered, result)
    if dist.get_rank() == 0:
        assert [r['rank'] for r in gathered] == list(range(8))
        assert all(r['shard_group_size'] == 8 for r in gathered)
        assert all(r['replica_group_size'] == 1 for r in gathered)
        assert all(not r['cuda_initialized'] for r in gathered)
        out = Path('submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/CPU_GROUP_TEST.json')
        out.write_text(json.dumps({'status':'GLOO_GROUP_PASS_NO_FSDP_MODEL','ranks':gathered},indent=2)+'\n')
        print('GLOO_GROUP_PASS_NO_FSDP_MODEL')
finally:
    dist.destroy_process_group()
