"""One authorized EXP-026/v3 load-only attempt; no reader, sampler or codec."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def current_rss_bytes() -> int:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    return 0


def mem_available_bytes() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    return 0


def main() -> int:
    import torch
    import torch.distributed as dist
    from solarwm.config.loader import load_config
    from solarwm.backends.minimax_h3.config import validate_h3_config
    from solarwm.backends.minimax_h3.fsdp import initialize_distributed, frozen_base_groups
    from solarwm.backends.minimax_h3 import runtime
    from solarwm.backends.minimax_h3.weights import load_initial_weights, canonical_lora_key
    from solarwm.runtime import Topology, rng_identity
    from solarwm.runtime.randomness import seed_process

    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--log-dir',type=Path,required=True)
    args=parser.parse_args()
    rank=int(os.environ['RANK'])
    local=int(os.environ['LOCAL_RANK'])
    world=int(os.environ['WORLD_SIZE'])
    if world!=4 or os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,6,7':
        raise RuntimeError('four-rank physical GPU0/1/6/7 contract differs')
    args.log_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    log_file=(args.log_dir/f'rank_{rank}.jsonl').open('a',buffering=1)
    def event(stage, **extra):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize(local)
        record={
            'stage':stage,'rank':rank,'local_rank':local,'pid':os.getpid(),
            'wall_unix':time.time(),'elapsed_sec':time.perf_counter()-started,
            'rss_bytes':current_rss_bytes(),
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'host_mem_available_bytes':mem_available_bytes(),
            'cuda_allocated_bytes':torch.cuda.memory_allocated(local) if torch.cuda.is_initialized() else 0,
            'cuda_reserved_bytes':torch.cuda.memory_reserved(local) if torch.cuda.is_initialized() else 0,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(local) if torch.cuda.is_initialized() else 0,
            **extra,
        }
        log_file.write(json.dumps(record,sort_keys=True)+'\n')
        if record['cuda_allocated_bytes']>40*2**30 or record['cuda_peak_allocated_bytes']>40*2**30:
            raise RuntimeError('per-rank allocated exceeded 40GiB')
        return record

    model=None
    lora=None
    try:
        expected=HERE/'patched_src/src/solarwm/backends/minimax_h3/runtime.py'
        if Path(runtime.__file__).resolve()!=expected.resolve():
            raise RuntimeError('load-only entry did not import independently patched runtime')
        auth=json.loads((ROOT/'submission/experiments/EXP-026_solar_stage2_reference/AUTHORIZATION_v3.json').read_text())
        if auth.get('plan_version')!=3 or auth.get('gpu_seconds')!=2400 or not auth.get('user_approved'):
            raise RuntimeError('EXP-026/v3 authorization contract differs')
        config=load_config(args.config).values
        validate_h3_config(config)
        if config['model'].get('cpu_sharded_load') is not True or config['model'].get('local_files_only') is not True:
            raise RuntimeError('CPU-deferred, local-only config required')
        if config['distributed']['world_size']!=4 or config['distributed']['sequence_parallel_size']!=4:
            raise RuntimeError('SP4/world4 config required')
        if config['validation']['sample_count']!=1:
            raise RuntimeError('one-case config required')
        torch.set_num_threads(1)
        topology=Topology.from_environ(4)
        event('pre_nccl',patched_runtime=str(runtime.__file__),checkpoint=str(config['checkpoint']['resume_from']))
        initialize_distributed(sp_size=4,local_rank=local)
        device=torch.device('cuda',local)
        device_properties=torch.cuda.get_device_properties(device)
        if device_properties.total_memory<40*2**30:
            raise RuntimeError('visible device smaller than expected L40')
        shard,replica=frozen_base_groups(4)
        if dist.get_world_size(shard)!=4 or dist.get_world_size(replica)!=1:
            raise RuntimeError('physical shard topology differs')
        identity=rng_identity('minimax_h3',int(config['data'].get('seed',42)),topology)
        seed_process(identity.model_init_seed)
        event('nccl_ready',gpu_name=device_properties.name,shard_world=4,replica_world=1,
              sp_world=4,dp_world=1)
        model,lora,base_receipt=runtime._prepare_h3_inference_model(
            config,device=device,topology=topology,stage='stage2',event_hook=event,
        )
        if base_receipt['strict_state_load'] != {
            'missing_keys':0,'unexpected_keys':0,'mismatched_shapes':0,'load_errors':0,
        }:
            raise RuntimeError('strict base load receipt differs')
        if lora.parameter_count!=2075394048 or len(lora.parameter_by_key)!=624:
            raise RuntimeError('LoRA rank384 descriptor count differs')
        if any(parameter.device.type!='cuda' for parameter in lora.parameters):
            raise RuntimeError('ignored LoRA is not on local CUDA after FSDP')
        event('model_sharded',base_parameter_count=base_receipt['parameter_inventory']['parameters'],
              base_dtype_inventory=base_receipt['parameter_inventory']['dtypes'],
              lora_parameter_count=lora.parameter_count,lora_tensor_count=len(lora.parameter_by_key))
        checkpoint=config['checkpoint']['resume_from']
        weight_label=load_initial_weights(
            {'path':checkpoint,'stage':'stage2','weight_source':'ema'},lora,
        )
        if weight_label!='stage2:ema:step=3900':
            raise RuntimeError(f'unexpected EMA receipt: {weight_label}')
        event('ema_loaded',weight_label=weight_label,
              ema_bytes=(Path(checkpoint)/'ema.pt').stat().st_size,
              ema_sha256_prior_verified='92e8f4b07aa92b0453a2f63954db1421caa3b8cac83ece706535db1a91493a05')
        # Independent descriptor and sparse numerical spot check. This never
        # gathers the 33B base or copies a whole adapter to one GPU.
        payload=torch.load(Path(checkpoint)/'ema.pt',map_location='cpu',mmap=True,weights_only=True)
        source={canonical_lora_key(key):value for key,value in payload['shadow'].items()}
        keys=sorted(lora.parameter_by_key)
        samples=[]
        for key in (keys[0],keys[len(keys)//2],keys[-1]):
            parameter=lora.parameter_by_key[key]
            reference=source[key]
            if tuple(parameter.shape)!=tuple(reference.shape) or parameter.dtype!=torch.bfloat16:
                raise RuntimeError(f'EMA spot-check shape/dtype mismatch: {key}')
            indices=sorted({0,parameter.numel()//2,parameter.numel()-1})
            for index in indices:
                actual=float(parameter.reshape(-1)[index].float().item())
                expected=float(reference.reshape(-1)[index].to(torch.bfloat16).float().item())
                if not (torch.isfinite(torch.tensor(actual)) and actual==expected):
                    raise RuntimeError(f'EMA spot-check numerical mismatch: {key}:{index}')
                samples.append({'key':key,'index':index,'value':actual})
        del payload,source
        event('spot_check_pass',spot_checks=samples)
        event('load_only_pass',forward_count=0,codec_count=0,update_count=0)
        return 0
    except BaseException as exc:
        event('failure',exception=repr(exc),traceback=traceback.format_exc())
        return 1
    finally:
        try:
            del model,lora
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.destroy_process_group()
            event('released')
        except BaseException as cleanup_exc:
            log_file.write(json.dumps({'stage':'cleanup_failure','exception':repr(cleanup_exc)})+'\n')
        log_file.close()

if __name__=='__main__':
    raise SystemExit(main())
