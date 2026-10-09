"""Run A on a single history, shared frozen backbone and exact role snapshots."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import field as f

FM=f.ROOT/'outputs/2026-10-08-10/stage1_fm_full_history_shift12_32/train_32/step_32'
AF=f.ROOT/'outputs/2026-10-08-10/stage1_shift12_duration64/train_32/step_32'
LATEST=f.p.CHECKPOINT
ROLES=('original','causal_original','causal_initializer','causal_initializer_diagonal',
       'causal_fm32','causal_anyflow32','causal_anyflow32_finite',
       'causal_anyflow128','causal_anyflow128_finite')


def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main(args):
    out=f.OUT/f'field_{args.history}.json'
    if out.exists():raise FileExistsError(out)
    result=dict(status='preflight',history=args.history,gpu=args.gpu,pid=os.getpid(),
        started_at=datetime.now().astimezone().isoformat(),records=[],roles=list(ROLES),
        scope='IDENTICAL full-prefix tokens/positions/noise/dual RGB anchor/time policy; shared SDPA backend',
        limitation='History ancestors and action-prefix states recomputed, not deployed persistent-KV equivalence. Shared current anchor also conditions recomputed past.',
        state_protocol='Explicit generated-endpoint interpolants from AnyFlow128, not captured solver states',
        checkpoints={'fm32':str(FM),'af32':str(AF),'af128':str(LATEST)})
    tick=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-tick
        tmp=out.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
    save()
    try:
        assert json.loads((f.OUT/'cpu_field.json').read_text())['status']=='passed'
        free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<36000:result.update(status='not_started_gpu_busy',free_MiB=free);return
        # Source/runtime/checkpoint manifests are per lane to avoid writer races.
        files=[f.OUT/n for n in ('field.py','run_field.py','check_field.py','cpu_field.json')]
        files += [f.PREVIOUS/'probe.py',f.PREVIOUS/'manifest.json']
        files += [d/n for d in (FM,AF,LATEST) for n in ('causal_adapter.pt','action_adapter.pt','stage1_lora.pt')]
        files += [d/'anyflow_adapter.pt' for d in (AF,LATEST)]
        files += [f.p.GENERATED/a/n for a in ('A','D') for n in ('conditioning.pt','cached_latents.pt')]
        manifest={str(p):f.p.sha(p) for p in files}
        (f.OUT/f'manifest_{args.history}.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result.update(status='loading',gpu_free_start_MiB=free,weight_reserve_GiB=14);save()
        torch.set_num_threads(4);torch.manual_seed(13)
        import infer as abot
        from causal.train_online_selfrollout import action_condition,move_tree
        from causal.pretrained_lora import load_adapter,load_action_residual
        from causal.stage1_lora import load_stage1_lora
        from causal.anyflow import load_anyflow
        from causal.h3_precision import configure_precision,validate_precision_checkpoint
        from causal.h3_cached import expand_packed_two_anchors,last_frame_image_anchor,slice_packed
        snapshots={name:torch.load(directory/'stage1_lora.pt',map_location='cpu',weights_only=True)
                   for name,directory in [('fm32',FM),('af32',AF),('af128',LATEST)]}
        for file in ('causal_adapter.pt','action_adapter.pt'):
            states=[torch.load(d/file,map_location='cpu',weights_only=True) for d in (FM,AF,LATEST)]
            first={k:v for k,v in states[0].items() if k!='metadata'}
            assert all(same(first,{k:v for k,v in s.items() if k!='metadata'}) for s in states[1:]),file
        aftime=torch.load(AF/'anyflow_adapter.pt',map_location='cpu',weights_only=True)
        latesttime=torch.load(LATEST/'anyflow_adapter.pt',map_location='cpu',weights_only=True)
        assert same(aftime['weights'],latesttime['weights']) and aftime['gate']==latesttime['gate']
        config_fm=snapshots['fm32']['metadata']['config'];config_af=snapshots['af32']['metadata']['config']
        allowed={'out_dir','resume_from','objective'}
        assert all(config_fm[k]==config_af[k] for k in config_fm if k not in allowed)
        result['checkpoint_pair_audit']=dict(fm_af32_common_config_equal=True,
            frozen_visual_action_equal_all_roles=True,frozen_target_time_equal_32_128=True,
            fm_af32_optimizer_updates=32,equal_updates_not_equal_training_compute=True,
            caution='These FM/AnyFlow checkpoints trained on teacher-generated clips; NOT experiment B real-video training')
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(f.ROOT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        meta=load_adapter(model,LATEST/'causal_adapter.pt','cuda:0')
        visual=[model.blocks[i].attn.qkv_proj for i in meta['block_indices']]
        action=load_action_residual(model,LATEST/'action_adapter.pt','cuda:0')['adapter']
        precision=configure_precision(model,'h3_fp32',native_transformer_dir=
            f.ROOT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        time_module,time_meta=load_anyflow(model,LATEST/'anyflow_adapter.pt','cuda:0')
        bank,bank_meta=load_stage1_lora(model,LATEST/'stage1_lora.pt',device='cuda:0')
        validate_precision_checkpoint(meta['metadata'],precision);validate_precision_checkpoint(bank_meta,precision)
        model.requires_grad_(False);action.requires_grad_(False)
        bank_ids={id(x) for x in bank.parameters()}
        frozen_versions=[(x,x._version) for x in [*model.parameters(),*action.parameters()] if id(x) not in bank_ids]
        conds={a:move_tree(torch.load(f.p.GENERATED/a/'conditioning.pt',map_location='cpu',weights_only=True),'cuda:0') for a in ('A','D')}
        cond=conds[args.history]
        for key in ('anchor','audio_noise','initial_noise','packed'):
            assert same(conds['A'][key],conds['D'][key]),key
        generated=torch.load(f.p.GENERATED/args.history/'cached_latents.pt',map_location='cuda:0',weights_only=True).float()
        noise=cond['initial_noise'].float();packed=expand_packed_two_anchors(cond['packed'],frame_rows=390)
        actions={a:action_condition(a,12,'cuda:0',torch.bfloat16) for a in ('A','D')}
        anchors={}
        pipe.load_models_to_device(['video_vae'])
        with torch.no_grad():
            for chunk in (1,2):
                tail=last_frame_image_anchor(pipe.video_vae,generated[:,:,:chunk*5],dtype=pipe.torch_dtype)
                anchors[chunk]=torch.cat((cond['anchor'],tail))
        pipe.load_models_to_device(['dit'])
        loaded_bank='af128'
        def role(name):
            nonlocal loaded_bank
            pure=name in ('original','causal_original')
            for module in visual:module.enabled=not pure
            action.enabled=not pure
            trained=name.startswith(('causal_fm32','causal_anyflow'))
            for module in bank.modules:module.enabled=trained
            desired='fm32' if name=='causal_fm32' else ('af128' if '128' in name else 'af32')
            if trained and loaded_bank!=desired:
                snapshot=snapshots[desired]
                assert list(bank.names)==snapshot['targets']
                for key,module in zip(bank.names,bank.modules):
                    for params,values in [(module.lora_A,snapshot['weights'][key]['A']),
                                          (module.lora_B,snapshot['weights'][key]['B'])]:
                        for dest,src in zip(params,values):dest.copy_(src)
                loaded_bank=desired
                assert all(torch.equal(param.cpu(),weight)
                    for key,m in zip(bank.names,bank.modules)
                    for ps,ws in [(m.lora_A,snapshot['weights'][key]['A']),(m.lora_B,snapshot['weights'][key]['B'])]
                    for param,weight in zip(ps,ws))
            use_time='anyflow' in name or name=='causal_initializer_diagonal'
            model.anyflow_conditioner=time_module if use_time else None
            return ('original' if name=='original' else 'causal'),(None if pure else action),use_time
        result.update(status='running',precision=precision,parameter_bank_switch='exact checkpoint tensors, no optimizer',
            generated_sha256=f.p.tensor_sha(generated),noise_sha256=f.p.tensor_sha(noise),
            anchor_sha256={str(k):f.p.tensor_sha(v) for k,v in anchors.items()});save()
        model_calls=0
        with torch.no_grad():
            for chunk in (1,2):
                start=chunk*5;stop=min(start+5,12);history=generated[:,:,:start]
                prompts={a:f.p.replace_current(cond['prompt_embeds'],conds[a]['prompt_embeds'],packed,start,stop) for a in ('A','D')}
                ac={}
                for a in ('A','D'):
                    ac[a]=actions[args.history][:stop].clone();ac[a][start:]=actions[a][start:stop]
                for sigma,target in zip(f.p.SIGMAS,f.p.TARGETS):
                    z=(1-sigma)*generated[:,:,start:stop]+sigma*noise[:,:,start:stop]
                    zhash=f.p.tensor_sha(z);outputs={};case_tick=time.perf_counter()
                    record=dict(chunk=chunk,sigma=sigma,finite_target=target,start=start,stop=stop,
                        current_sha256=zhash,history_sha256=f.p.tensor_sha(history),
                        token_count=int(slice_packed(packed,0,stop,390)['seq_len']),
                        input_scope_identical_all_roles=True,comparisons={})
                    for name in ROLES:
                        mode,adapter,use_time=role(name)
                        r=target if name.endswith('_finite') else sigma
                        outputs[name]=[f.forward(model,z,history,cond,prompts[a],anchors[chunk],packed,sigma,
                            mode=mode,actions=ac[a],adapter=adapter,target=r if use_time else None).cpu()
                            for a in ('A','D')]
                        model_calls+=2
                        assert all(torch.isfinite(v).all() for v in outputs[name])
                        teacher=outputs['original']
                        record['comparisons'][name]=dict(
                            absolute_velocity={a:f.p.compare(outputs[name][i],teacher[i]) for i,a in enumerate(('A','D'))},
                            action_delta=f.p.delta_metrics(outputs[name],teacher),
                            semantics='finite interval average; descriptive only' if name.endswith('_finite') else 'instantaneous')
                    if sigma==f.p.SIGMAS[1]:
                        role('original')
                        native=[f.p.teacher_forward(model,z,history,cond,prompts[a],anchors[chunk],packed,sigma,matched=True).cpu() for a in ('A','D')]
                        model_calls+=2
                        record['backend_control']=dict(
                            delta=f.p.delta_metrics(outputs['original'],native),
                            absolute_rmse=[f.p.rms(v-w) for v,w in zip(outputs['original'],native)])
                    record['initializer_diagonal_preservation']=dict(
                        velocity_rmse=[f.p.rms(x-y) for x,y in zip(outputs['causal_initializer_diagonal'],outputs['causal_initializer'])],
                        delta=f.p.delta_metrics(outputs['causal_initializer_diagonal'],outputs['causal_initializer']))
                    record['fm32_vs_anyflow32']=f.p.delta_metrics(outputs['causal_anyflow32'],outputs['causal_fm32'])
                    record['anyflow128_vs32']=f.p.delta_metrics(outputs['causal_anyflow128'],outputs['causal_anyflow32'])
                    assert f.p.tensor_sha(z)==zhash
                    assert all(x._version==v for x,v in frozen_versions)
                    # Retain small action deltas locally for further analysis, not in submission.
                    tensors=f.OUT/'delta_tensors';tensors.mkdir(exist_ok=True)
                    tensor_file=tensors/f'{args.history}_chunk{chunk}_sigma{sigma:.6f}.pt'
                    torch.save({name:xs[0]-xs[1] for name,xs in outputs.items()},tensor_file)
                    record.update(wall_seconds=time.perf_counter()-case_tick,delta_tensor_sha256=f.p.sha(tensor_file))
                    result['records'].append(record);result['model_forwards']=model_calls;save()
                    print(json.dumps(dict(chunk=chunk,sigma=sigma,delta_cosines={n:round(v['action_delta']['cosine'],6) for n,v in record['comparisons'].items()})),flush=True)
        result.update(status='complete',model_forwards=model_calls,frozen_base_versions_unchanged=True,
            gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--history',choices=['A','D'],required=True);args=ap.parse_args()
    f.setup(args.gpu)
    import torch
    main(args)
