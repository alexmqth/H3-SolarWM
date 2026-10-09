"""P0 first; stop on equivalence failure. Fixed-state inference only, no videos."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
H3=BASE.parents[2]
SOURCE=H3/'outputs/2026-10-09-22/chunk_partition_cb'
RT=(SOURCE/'runtime').resolve()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from audit_core import forward,prefill,tensor_hash,cache_hash,errors,delta_metrics
    torch.set_num_threads(4);torch.manual_seed(13)
    protocol=json.loads((BASE/'protocol.json').read_text())
    for rel,digest in protocol['source_sha256'].items(): assert sha(BASE/rel)==digest,rel
    for rel,digest in protocol['inputs_sha256'].items(): assert sha(SOURCE/rel)==digest,rel
    out=BASE/args.output;out.mkdir(exist_ok=False)
    result=dict(status='loading',started_at=datetime.now().astimezone().isoformat(),
        gpu=args.gpu,pid=os.getpid(),optimizer_updates=0,generated_videos=0,
        model_forwards=0,prefill_forwards=0,current_forwards=0,other_forwards=0,
        P0=[],P1=[],P2=[],controls=[],protocol_sha256=sha(BASE/'protocol.json'),
        source_sha256=protocol['source_sha256'])
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        tmp=out/'evaluation.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'evaluation.json')
    def call(fn,*a,category='current',**kw):
        value=fn(*a,**kw);result['model_forwards']+=1;result[category+'_forwards']+=1;return value
    def move(x):
        if torch.is_tensor(x):return x.to('cuda:0')
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    save()
    try:
        data={a:torch.load(SOURCE/f'source_coarse/inputs/parking_{a}.pt',map_location='cpu',weights_only=True) for a in 'AD'}
        for field in ('initial_noise','audio_noise','anchor'):
            assert torch.equal(data['A'][field],data['D'][field]),field
        packed=move(data['A']['packed']);spans=packed['action_text_spans_local']
        initial=data['A']['initial_noise'].to('cuda:0',torch.float32)
        audio=data['A']['audio_noise'].to('cuda:0');anchor=data['A']['anchor'].to('cuda:0')
        bank={}
        for ref in 'AD':
            h=torch.load(SOURCE/f'source_coarse/coarse_A/window0_{ref}.pt',map_location='cpu',weights_only=True).to('cuda:0')
            z=torch.load(SOURCE/f'C_{ref}/sigma_noised_{ref}_endpoint.pt',map_location='cpu',weights_only=True).to('cuda:0')
            bank[ref]=dict(history=h,endpoint=z,history_sha256=tensor_hash(h),endpoint_sha256=tensor_hash(z))
        result['input_hashes']={ref:{k:v for k,v in b.items() if k.endswith('sha256')} for ref,b in bank.items()}
        result['input_hashes'].update(anchor=tensor_hash(anchor),audio=tensor_hash(audio),noise=tensor_hash(initial))
        def cond(ref,action,stop=17,past_override=None):
            text=data[ref]['pairs'][0]['prompts'][ref].clone()
            if past_override is not None:
                donor=data[past_override]['pairs'][0]['prompts'][past_override]
                for lo,hi in spans[:12]:text[lo:hi]=donor[lo:hi]
            donor=data[action]['pairs'][0]['prompts'][action]
            for lo,hi in spans[12:stop]:text[lo:hi]=donor[lo:hi]
            layout,text=visible_inputs(packed,text.to('cuda:0'),stop,390)
            return dict(packed=layout,prompt=text,anchor=anchor,audio=audio)
        pipe=abot.load_pipeline('cuda:0')
        released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.eval().requires_grad_(False)
        result['released_LoRA_sha256']=sha(released)
        result['precision']=configure_precision(model,'h3_fp32',
            native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit']);torch.cuda.synchronize()
        result['load_seconds']=time.perf_counter()-began
        versions=[(p,p._version) for p in model.parameters()]
        torch.cuda.reset_peak_memory_stats()
        sigmas=protocol['sigmas'];saved={}
        with torch.no_grad():
            # Exact replay of the recorded first12 solver input; no new rollout.
            probe=torch.load(SOURCE/'source_coarse/coarse_A/window0_solver_states.pt',map_location='cpu',weights_only=True)[1]
            pred,_=call(forward,model,probe['state'].to('cuda:0'),history=initial[:,:,:0],
                sigma=probe['sigma'],kind='R1',category='other',**cond('A','A',12))
            err=errors(pred,probe['velocity_A'])
            result['controls'].append(dict(name='first12_frozen_source_replay',errors=err))
            assert err['relative_rms']<1e-6,err
            result['status']='P0_matching_context';save()
            for ref in 'AD':
                h=bank[ref]['history']
                for si,sigma in enumerate(sigmas):
                    state=(1-sigma)*bank[ref]['endpoint']+sigma*initial[:,:,12:17]
                    cache=call(prefill,model,h,sigma=sigma,category='prefill',**cond(ref,ref,12))
                    digest=cache_hash(cache);pairs={};records=[]
                    for action in 'AD':
                        conds=cond(ref,action)
                        v,control=call(forward,model,state,history=h,sigma=sigma,kind='R2',compare_cache=cache,**conds)
                        cached,_=call(forward,model,state,history=h,sigma=sigma,kind='cached',cache=cache,**conds)
                        record=dict(action=action,velocity=errors(cached,v),layers=control.layer_errors)
                        records.append(record);pairs[action]=v.cpu();saved[ref,si,action]=v.cpu()
                        torch.save(dict(R2=v.cpu(),R3_matched=cached.cpu()),out/f'P0_{ref}_{si}_{action}.pt')
                    unchanged=cache_hash(cache)==digest
                    maxkv=max(e[k]['relative_rms'] for rec in records for e in rec['layers'] for k in ('key','value'))
                    maxv=max(r['velocity']['relative_rms'] for r in records)
                    pass_gate=unchanged and maxkv<protocol['P0_relative_tolerance'] and maxv<protocol['P0_relative_tolerance']
                    row=dict(history=ref,sigma=sigma,state_sha256=tensor_hash(state),cache_sha256=digest,
                        CPU_KV_MiB=cache.nbytes/2**20,cache_unchanged=unchanged,records=records,
                        max_KV_relative_rms=maxkv,max_velocity_relative_rms=maxv,passed=pass_gate)
                    result['P0'].append(row);save()
                    print(f'P0 {ref} sigma={sigma}: KV={maxkv:.8g}, velocity={maxv:.8g}, pass={pass_gate}',flush=True)
                    cache.clear();del cache
                    if not pass_gate:
                        result['status']='P0_mismatch_stop_before_P1';save();return
            result['P0_passed']=True;result['status']='P1_factorization';save()
            for ref in 'AD':
                h=bank[ref]['history']
                cache=call(prefill,model,h,sigma=0,category='prefill',**cond(ref,ref,12))
                digest=cache_hash(cache)
                for si,sigma in enumerate(sigmas):
                    state=(1-sigma)*bank[ref]['endpoint']+sigma*initial[:,:,12:17]
                    pairs={'R2':{a:saved[ref,si,a] for a in 'AD'}}
                    roles=[('R1','R1','own'),('R1P','R1P','own'),('R3','cached','own'),
                           ('R4_within','cached','within'),('R4_full','cached','full')]
                    if si==1:roles.append(('R4_cross','cached','cross'))
                    for name,kind,route in roles:
                        pair={}
                        for action in 'AD':
                            v,_=call(forward,model,state,history=h,sigma=sigma,kind=kind,route=route,
                                cache=cache,**cond(ref,action))
                            pair[action]=v.cpu()
                        pairs[name]=pair
                        print(f'P1 {ref} sigma={sigma}: {name}',flush=True)
                    comparisons={f'{x}_vs_{y}':delta_metrics(pairs[x],pairs[y]) for x,y in
                        [('R1P','R1'),('R2','R1P'),('R3','R2'),('R4_within','R3'),('R4_full','R3'),('R4_full','R4_within')]}
                    if si==1:comparisons['R4_cross_vs_R3']=delta_metrics(pairs['R4_cross'],pairs['R3'])
                    row=dict(history=ref,sigma=sigma,state_sha256=tensor_hash(state),comparisons=comparisons,
                        vs_Original={name:delta_metrics(pair,pairs['R1']) for name,pair in pairs.items()})
                    result['P1'].append(row)
                    torch.save(pairs,out/f'P1_{ref}_{si}.pt');save()
                    if si==1:
                        nh=(1-sigma)*h+sigma*initial[:,:,:12]
                        npairs={}
                        for kind in ('R1','R2'):
                            pair={}
                            for action in 'AD':
                                v,_=call(forward,model,state,history=nh,sigma=sigma,kind=kind,history_noised=True,**cond(ref,action))
                                pair[action]=v.cpu()
                            npairs[kind]=pair
                        result['P2'].append(dict(history=ref,sigma=sigma,
                            R2N_vs_R1N=delta_metrics(npairs['R2'],npairs['R1']),
                            R1clean_vs_R1N=delta_metrics(pairs['R1'],npairs['R1']),
                            note='R2N/R1N share identical noised history; clean/N differs in history values and time'))
                        torch.save(npairs,out/f'P2_N_{ref}.pt');save()
                assert cache_hash(cache)==digest
                cache.clear();del cache
            assert all(tensor_hash(b['history'])==b['history_sha256'] for b in bank.values())
            assert all(p._version==v for p,v in versions)
            result['frozen_weights_and_history_unchanged']=True
        result['status']='complete_fixed_state_only'
    except BaseException as exc:
        result['status']='failed';result['error']=repr(exc);raise
    finally:
        result['GPU_peak_MiB']=torch.cuda.max_memory_allocated()/2**20
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--output',default='run_01')
    a=parser.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need an idle GPU with >=40000 MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',TOKENIZERS_PARALLELISM='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
