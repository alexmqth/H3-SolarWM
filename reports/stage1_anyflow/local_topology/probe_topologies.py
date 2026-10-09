"""E1 same-state geometry on actual T2 solver states, Original weights only.

C0/C1/T1 each rebuild their own clean historical video KV. Within each
action pair only current annotations change, with the same immutable cache.
T2 recomputes the same visible raw history. It is also R-prefix, not an
independently improved model. No teacher outputs from different states are
subtracted. This probe alone cannot accept video quality or action behavior.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
RT=BASE/'runtime'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs,window_forward,grounded_prefix
    from causal.h3_cached import H3ChunkCache,chunk_forward
    torch.set_num_threads(4);torch.manual_seed(13)
    positive=BASE/f'positive_{args.history}'
    accepted=json.loads((BASE/'positive_review.json').read_text())
    if not accepted.get('local_reference_usable'):
        raise RuntimeError('Review/calibrate the local positive control before geometry expansion')
    receipt=json.loads((positive/'evaluation.json').read_text())
    if receipt['status']!='complete_pending_visual_review':raise RuntimeError('Positive run incomplete')
    if accepted['evaluation_sha256'][args.history]!=sha(positive/'evaluation.json'):
        raise RuntimeError('Positive review and source receipt disagree')
    sources=json.loads((BASE/'runtime_manifest.json').read_text())
    for rel,h in sources.items():assert sha(RT/rel)==h,rel
    saved=torch.load(BASE/f'inputs/parking_{args.history}.pt',map_location='cpu',weights_only=True)
    out=BASE/f'geometry_{args.history}';out.mkdir(exist_ok=False)
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    result=dict(status='loading',pid=os.getpid(),start_ticks=stat[19],gpu=args.gpu,
        started_at=datetime.now().astimezone().isoformat(),scope=__doc__,
        source_sha256=sha(__file__),runtime_manifest_sha256=sha(BASE/'runtime_manifest.json'),
        positive_sha256=sha(positive/'evaluation.json'),history_reference=args.history,
        optimizer_updates=0,records=[],model_forwards=0,stage1_accepted=False)
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        tmp=out/'probe.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'probe.json')
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def digest(cache):
        h=hashlib.sha256()
        for layer,es in sorted(cache.layers.items()):
            for e in es:
                h.update(str((layer,e.index)).encode())
                for x in (e.key,e.value,e.rope):h.update(th(x).encode())
        return h.hexdigest()
    def versions(cache):return [(x,x._version,x.data_ptr()) for es in cache.layers.values() for e in es for x in (e.key,e.value,e.rope)]
    def intact(items):return all(x._version==v and x.data_ptr()==p for x,v,p in items)
    def metrics(a,b):
        x,y=a.double().flatten(),b.double().flatten();xn=float(x.norm());yn=float(y.norm())
        return dict(cosine=float(torch.dot(x,y)/(x.norm()*y.norm())) if min(xn,yn)>1e-12 else None,
            student_norm=xn,reference_norm=yn,norm_ratio=xn/max(yn,1e-30),
            relative_error=float((x-y).norm())/max(yn,1e-30),
            rmse=float((x-y).square().mean().sqrt()))
    def move(x):
        if torch.is_tensor(x):return x.to('cuda:0')
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    def scope(role):return nullcontext() if role=='C0' else grounded_prefix(history=role=='T1')
    save()
    try:
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(RT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit'])
        parameter_versions=[(p,p._version) for p in model.parameters()]
        full=move(saved['packed']);audio=saved['audio_noise'].to('cuda:0')
        history=saved['reference_latents'].to('cuda:0',torch.float32)
        anchors=[a.to('cuda:0') for a in saved['anchors']]
        def condition(i,a):
            pair=saved['pairs'][i]
            packed,prompt=visible_inputs(full,pair['prompts'][a].to('cuda:0'),pair['stop'],390)
            return dict(full_packed=packed,prompt=prompt,anchor=anchors[i],audio=audio,chunk_frames=5,anchor_slot=1)
        def cached(role,state,i,a,sigma,cache,commit=False):
            with scope(role):
                y=chunk_forward(model,state,index=i,cache=cache,sigma=sigma,commit=commit,
                    action_prefix_mode='causal' if role=='C0' else 'own',action_feedback=True,
                    anchor_frame_index=i*5-1 if i else None,**condition(i,a))
            result['model_forwards']+=1;return y.float().cpu()
        def window(state,i,a,sigma):
            y=window_forward(model,state,index=i,history=history[:,:,:i*5],sigma=sigma,
                history_chunks=5,**condition(i,a))
            result['model_forwards']+=1;return y.float().cpu()
        caches={role:H3ChunkCache(5,'cpu') for role in ('C0','C1','T1')}
        result['status']='running';save()
        with torch.no_grad():
            for i in range(3):
                paths=positive/f'chunk{i}_solver_states.pt'
                probes=torch.load(paths,map_location='cpu',weights_only=True)
                assert len(probes)==3 and [s['step'] for s in probes]==[0,15,27]
                before={r:digest(c) for r,c in caches.items()}
                immutable={r:versions(c) for r,c in caches.items()}
                for point in probes:
                    state=point['state'].to('cuda:0');sigma=point['sigma']
                    assert th(state)==point['state_sha256']
                    assert th(history[:,:,:i*5])==point['history_sha256']
                    teacher=[window(state,i,a,sigma) for a in 'AD']
                    repeat_error=float((teacher[0]-point['velocity_A'].float()).abs().max())
                    # Same source/weights/inputs/backend. A stored repeat is
                    # an actual guard, not merely matching scalar norms.
                    if repeat_error!=0:raise RuntimeError(f'Original solver replay mismatch: {repeat_error}')
                    preds={role:[cached(role,state,i,a,sigma,caches[role]) for a in 'AD'] for role in caches}
                    record=dict(chunk=i,step=point['step'],sigma=sigma,state_kind=point['state_kind'],
                        state_sha256=th(state),history_sha256=point['history_sha256'],
                        input_file_sha256=sha(paths),teacher_replay_max_abs=repeat_error,
                        reference_prediction_sha256={a:th(v) for a,v in zip('AD',teacher)},
                        roles={role:dict(whole={a:metrics(v,teacher[j]) for j,(a,v) in enumerate(zip('AD',pred))},
                            delta=metrics(pred[0]-pred[1],teacher[0]-teacher[1]),
                            per_frame_delta=[metrics((pred[0]-pred[1])[:,:,f],(teacher[0]-teacher[1])[:,:,f])
                                for f in range(state.shape[2])]) for role,pred in preds.items()})
                    if point['step']==15:
                        # Explicit ancestor replay at each held history, not
                        # full-prefix bidirectional recomputation masquerading
                        # as immutable-cache replay.
                        replay={}
                        for role in caches:
                            fresh=H3ChunkCache(5,'cpu')
                            for old in range(i):
                                cached(role,history[:,:,old*5:old*5+5],old,args.history,0.,fresh,True)
                            assert digest(fresh)==before[role]
                            repeated=cached(role,state,i,'A',sigma,fresh)
                            error=float((repeated-preds[role][0]).abs().max())
                            assert error==0,role
                            replay[role]=error;fresh.clear()
                        record['ancestry_replay_max_abs']=replay
                    assert all(intact(immutable[r]) for r in caches)
                    result['records'].append(record);save()
                    print(json.dumps(dict(history=args.history,chunk=i,step=point['step'],
                        delta_cosines={r:record['roles'][r]['delta']['cosine'] for r in caches})),flush=True)
                for role,cache in caches.items():
                    assert digest(cache)==before[role]
                    cached(role,history[:,:,i*5:min(i*5+5,12)],i,args.history,0.,cache,True)
                result.setdefault('cache_checkpoints',[]).append(dict(chunk=i,
                    readonly_during_AD=True,cpu_KV_MiB={r:c.nbytes/2**20 for r,c in caches.items()},
                    per_layer_commits={r:c.commits for r,c in caches.items()}));save()
            assert all(p._version==v for p,v in parameter_versions)
        result.update(status='complete',parameter_versions_unchanged=True,
            GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            future_content_excluded_before_model=True,
            limitation='Fixed parking references and 30-step T2 A trajectories. Delta direction is not a video-quality or free-rollout gate.')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--history',choices=['A','D'],required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
