"""48 fixed-state forwards: actual observed-transition fit versus wrong-action inflation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'


def main(args):
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_transition import transition_forward
    from causal.local_topology import visible_inputs
    from eval_common import sha,tensor_hash as th,move,load_bank,terminal
    torch.set_num_threads(4);torch.manual_seed(13)
    launch=json.loads((BASE/'heldout_fit_launch.json').read_text());assert sha(__file__)==launch['source_sha256']
    spec=json.loads((BASE/'evaluation_protocol.json').read_text());assert sha(BASE/'evaluation_protocol.json')==launch['evaluation_protocol_sha256']
    for rel,expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==expected
    for arm in ['fm_only','fm_action']:
        terminal(json.loads((BASE/f'train_{arm}/training.json').read_text()))
        for step in (0,4):assert sha(BASE/f'train_{arm}/step_{step:02d}/action_lora.pt')==spec['checkpoints'][arm][str(step)]
    enc=BASE/'encoded_validation/encoding.json';assert sha(enc)==spec['GT_encoding_sha256']
    out=BASE/'heldout_fit';out.mkdir(exist_ok=False);begin=time.perf_counter()
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=args.gpu,
        launch_sha256=sha(BASE/'heldout_fit_launch.json'),records=[],forwards=0,optimizer_updates=0,
        interpretation='Same actual observed target u for both labels. Negative has no counterfactual video truth. Validation coefficient tuning prohibited.')
    def save():
        result['wall_seconds']=time.perf_counter()-begin
        p=out/'fit.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'fit.json')
    save()
    try:
        pipe=abot.load_pipeline('cuda:0');released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        precision=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit']);result['precision']=precision;torch.cuda.reset_peak_memory_stats()
        cases=[]
        for rec in json.loads(enc.read_text())['clips']:
            assert sha(rec['encoded_path'])==rec['sha256']
            d=torch.load(rec['encoded_path'],map_location='cpu',weights_only=True)
            assert d['source']['split']=='validation' and '24' in d['negatives']
            cases.append(d)
        with torch.no_grad():
            for role in ['zero','fm_only','fm_action']:
                arm='fm_only' if role=='zero' else role;step=0 if role=='zero' else 4
                bank,meta=load_bank(model,BASE/f'train_{arm}/step_{step:02d}/action_lora.pt',precision=precision,check_original=role=='zero')
                versions=[(p,p._version) for p in list(model.parameters())+bank]
                result['status']='evaluating';save()
                for d in cases:
                    history=d['safe_prefixes'][24].to('cuda:0',torch.float32)
                    noise=d['initial_noise'].to('cuda:0',torch.float32)
                    clean=d['clean_latents'][:,:,24:36].to('cuda:0',torch.float32)
                    target=noise[:,:,24:36]-clean;hh,nh=th(history),th(noise)
                    conditions={}
                    for name,c in [('positive',d['positive']),('negative',d['negatives']['24'])]:
                        p,t=visible_inputs(move(c['packed']),c['prompt_embeds'].to('cuda:0'),36,390)
                        conditions[name]=dict(full_packed=p,prompt=t)
                    for sigma in spec['heldout_fit_diagnostic']['sigmas']:
                        state=(1-sigma)*clean+sigma*noise[:,:,24:36];fields={};errors={};per_frame={}
                        torch.cuda.synchronize();tick=time.perf_counter()
                        for branch in ['positive','negative']:
                            field=transition_forward(model,state,history=history,history_noise=noise[:,:,:24],
                                anchor=d['anchor'].to('cuda:0'),audio=d['audio_noise'].to('cuda:0'),sigma=sigma,index=2,**conditions[branch])
                            result['forwards']+=1;assert torch.isfinite(field).all()
                            e=(field.float()-target).square();errors[branch]=float(e.mean());per_frame[branch]=e.mean((0,1,3,4)).cpu().tolist()
                            fields[branch]=field
                        delta_rms=float((fields['positive']-fields['negative']).float().square().mean().sqrt());torch.cuda.synchronize()
                        result['records'].append(dict(role=role,clip_id=d['source']['clip_id'],sigma=sigma,history_latents=24,
                            positive_FM=errors['positive'],negative_FM=errors['negative'],gap=errors['negative']-errors['positive'],
                            per_latent_FM=per_frame,same_state_action_delta_rms=delta_rms,history_sha256=hh,noise_sha256=nh,
                            state_sha256=th(state),target_sha256=th(target),seconds=time.perf_counter()-tick))
                        save();print(json.dumps({k:result['records'][-1][k] for k in ['role','clip_id','sigma','positive_FM','negative_FM','gap']}),flush=True)
                        del fields,field,e
                    assert th(history)==hh and th(noise)==nh
                    del history,noise,clean,target,conditions
                assert all(p._version==v for p,v in versions)
            assert result['forwards']==48 and len(result['records'])==24
            for d in cases:
                for sigma in spec['heldout_fit_diagnostic']['sigmas']:
                    rows=[r for r in result['records'] if r['clip_id']==d['source']['clip_id'] and r['sigma']==sigma]
                    assert len(rows)==3
                    for field in ['history_sha256','noise_sha256','state_sha256','target_sha256']:assert len({r[field] for r in rows})==1
        result.update(status='complete',state_pairing_verified=True,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
