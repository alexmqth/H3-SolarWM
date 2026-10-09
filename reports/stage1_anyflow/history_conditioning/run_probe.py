"""28 forwards per reference: C/N same-state action effects, never teacher fidelity."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime';OLD=BASE/'source_coarse'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from history_conditioning import history_forward
    torch.set_num_threads(4);torch.manual_seed(13)
    cpu=json.loads((BASE/'cpu_receipt.json').read_text());assert cpu['status']=='passed'
    assert sha(BASE/'cpu_tests.log')==cpu['log_sha256']
    protocol=json.loads((BASE/'protocol.json').read_text())
    for rel,expected in protocol['sources'].items():assert sha(BASE/rel)==expected,rel
    for rel,expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==expected,rel
    inputs=OLD/f'inputs/parking_{args.history}.pt'
    prep=json.loads((OLD/'inputs/preparation.json').read_text())
    assert sha(inputs)==prep['files'][inputs.name]
    data=torch.load(inputs,map_location='cpu',weights_only=True)
    old_receipt=json.loads((OLD/f'coarse_{args.history}/evaluation.json').read_text())
    assert old_receipt['status']=='complete_pending_visual_review'
    probes={i:torch.load(OLD/f'coarse_{args.history}/window{i}_solver_states.pt',map_location='cpu',weights_only=True) for i in range(3)}
    out=BASE/f'probe_{args.history}';out.mkdir(exist_ok=False)
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),start_ticks=ticks,
                gpu=args.gpu,history_reference=args.history,history_kind=data['history_kind'],scope=__doc__,
                source_sha256=sha(__file__),protocol_sha256=sha(BASE/'protocol.json'),
                input_sha256=sha(inputs),source_evaluation_sha256=sha(OLD/f'coarse_{args.history}/evaluation.json'),
                source_state_sha256={str(i):sha(OLD/f'coarse_{args.history}/window{i}_solver_states.pt') for i in range(3)},
                diagnostic_forwards=0,optimizer_updates=0,CPU_KV_MiB=0,records=[],preflight={},repeat_checks=[],
                delta_reference_note='C is failed clean-history control, not a validated local teacher. C/N cosine measures change, not action correctness.',
                effects_accepted=False,stage1_accepted=False)
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        p=out/'probe.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'probe.json')
    def th(value):
        x=value.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def move(value):
        if torch.is_tensor(value):return value.to('cuda:0')
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        return value
    def metrics(a,b):
        a=a.double().flatten();b=b.double().flatten();diff=a-b
        na=float(a.norm());nb=float(b.norm())
        return dict(max_abs=float(diff.abs().max()),difference_rms=float(diff.square().mean().sqrt()),
                    a_rms=float(a.square().mean().sqrt()),b_rms=float(b.square().mean().sqrt()),
                    cosine=float(torch.dot(a,b)/(na*nb)) if min(na,nb)>1e-10 else None,
                    norm_ratio_a_over_b=na/nb if nb>1e-10 else None)
    save()
    try:
        pipe=abot.load_pipeline('cuda:0');released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        result['precision']=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        result['released_LoRA_sha256']=sha(released)
        pipe.load_models_to_device(['dit']);torch.cuda.synchronize()
        result['load_seconds']=time.perf_counter()-began
        history=data['reference_latents'].to('cuda:0',torch.float32)
        initial=data['initial_noise'].to('cuda:0',torch.float32)
        audio=data['audio_noise'].to('cuda:0');anchor=data['anchor'].to('cuda:0');packed=move(data['packed'])
        versions=[(p,p._version) for p in model.parameters()]
        initial_hash,history_hash=th(initial),th(history)
        result.update(full_history_sha256=history_hash,full_noise_sha256=initial_hash,audio_sha256=th(audio),anchor_sha256=th(anchor))
        conditions={}
        for i in range(3):
            for a in 'AD':
                layout,text=visible_inputs(packed,data['pairs'][i]['prompts'][a].to('cuda:0'),(i+1)*12,390)
                oldrow=next(r for r in old_receipt['records'] if r['window']==i and r['action']==a)
                assert th(text)==oldrow['prompt_sha256']
                assert th(layout['img_position_ids'])==oldrow['layout_positions_sha256']
                conditions[i,a]=dict(full_packed=layout,prompt=text,anchor=anchor,audio=audio,chunk_frames=12,anchor_slot=0)
        def predict(state,i,a,sigma,mode):
            output=history_forward(model,state,history=history[:,:,:i*12],history_noise=initial[:,:,:i*12],
                                   mode=mode,sigma=sigma,index=i,history_chunks=5,**conditions[i,a])
            result['diagnostic_forwards']+=1
            if result['diagnostic_forwards']>28:raise RuntimeError('Exceeded per-reference forward budget')
            value=output.detach().cpu().float();del output
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite velocity')
            save()
            return value
        torch.cuda.reset_peak_memory_stats()
        outputs=[]
        with torch.no_grad():
            first=next(p for p in probes[0] if p['step']==15)
            state=first['state'].to('cuda:0')
            c=predict(state,0,'A',first['sigma'],'C');n=predict(state,0,'A',first['sigma'],'N')
            result['preflight']=dict(first_window_identity=metrics(n,c),first_window_saved_C_replay=metrics(c,first['velocity_A']))
            assert torch.equal(c,n),'No-history identity failed'
            replay=result['preflight']['first_window_saved_C_replay']
            assert replay['difference_rms']<=protocol['saved_replay_relative_rms_max']*max(replay['b_rms'],1e-12),'Saved first-window replay mismatch'
            result['status']='probing';save()
            for i in (1,2):
                assert [p['step'] for p in probes[i]]==[0,15,27]
                for point in probes[i]:
                    assert point['history_sha256']==th(history[:,:,:i*12])
                    state=point['state'].to('cuda:0');assert th(state)==point['state_sha256']
                    velocities={}
                    for mode in ('C','N'):
                        for a in 'AD':velocities[mode+a]=predict(state,i,a,point['sigma'],mode)
                    saved_replay=metrics(velocities['CA'],point['velocity_A'])
                    assert saved_replay['difference_rms']<=protocol['saved_replay_relative_rms_max']*max(saved_replay['b_rms'],1e-12),'Saved C replay mismatch'
                    if point['step']==15:
                        repeated=predict(state,i,'A',point['sigma'],'C')
                        rec=dict(window=i,step=15,comparison=metrics(repeated,velocities['CA']))
                        result['repeat_checks'].append(rec)
                        assert torch.equal(repeated,velocities['CA']),'Nonzero repeat floor'
                    dc=velocities['CA']-velocities['CD'];dn=velocities['NA']-velocities['ND']
                    row=dict(window=i,step=point['step'],sigma=point['sigma'],current_state_sha256=th(state),
                             history_sha256=point['history_sha256'],history_noise_sha256=th(initial[:,:,:i*12]),
                             state_kind=point['state_kind'],current_action_only=True,saved_C_replay=saved_replay,
                             delta_N_vs_C=metrics(dn,dc),whole_N_vs_C_A=metrics(velocities['NA'],velocities['CA']),
                             per_latent_delta_C_rms=dc.double().square().mean((0,1,3,4)).sqrt().tolist(),
                             per_latent_delta_N_rms=dn.double().square().mean((0,1,3,4)).sqrt().tolist())
                    result['records'].append(row)
                    outputs.append(dict(window=i,step=point['step'],sigma=point['sigma'],state_sha256=point['state_sha256'],**velocities))
                    assert th(state)==point['state_sha256'] and th(history)==history_hash and th(initial)==initial_hash
                    save();print(f'[ref{args.history}] window{i} step{point["step"]} delta N/C ratio={row["delta_N_vs_C"]["norm_ratio_a_over_b"]:.6g} C-replay={saved_replay["max_abs"]:.6g}',flush=True)
                    del velocities,state,dc,dn
            assert result['diagnostic_forwards']==28
            assert all(p._version==v for p,v in versions)
            torch.save(outputs,out/'probe_fields.pt')
        result.update(status='complete',parameter_versions_unchanged=True,history_noise_unchanged=True,
                      GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,fields_sha256=sha(out/'probe_fields.pt'),
                      measurable_change=any(r['delta_N_vs_C']['difference_rms']>protocol['measurable_delta_rms_min'] for r in result['records']))
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True);p.add_argument('--history',choices=['A','D'],required=True);a=p.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with>=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
