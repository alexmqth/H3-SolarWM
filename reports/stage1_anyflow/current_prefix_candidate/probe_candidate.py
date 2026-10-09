"""Frozen-weight matched-state current-prefix candidate; no optimization.

Compare deployed current causal cache, isolated own/current-prefix cache and
Original bidirectional teacher on exactly the same step00-generated states.
Rebuild independent caches with each routing rule. The common prefix is private
to each chunk; it never modifies already committed past representations.
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

BASE = Path(__file__).resolve().parent
BRIDGE = BASE.parents[1] / '2026-10-08-18/stage1_real_abot_fm'
RT = BRIDGE / 'runtime'
MANIFEST = BRIDGE.parents[2] / 'data/abot_bridge/encoded_manifest.json'
SIGMAS = (.9395404663085938, .6894410400390625, .24078089904785155)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import torch
    import infer as abot
    import geometry_primitives as g
    from causal.h3_cached import H3ChunkCache, chunk_forward, last_frame_image_anchor
    from causal.h3_precision import configure_precision
    from causal.pretrained_lora import load_adapter
    from causal.stage1_lora import load_stage1_lora
    from current_prefix import current_prefix_feedback
    torch.set_num_threads(4); torch.manual_seed(13)
    preflight = json.loads((BASE/'cpu_preflight.json').read_text())
    assert preflight['status'] == 'passed'
    assert all(sha(BASE/name) == h for name,h in preflight['sources'].items())
    dest = BASE/'probe.json'
    if dest.exists(): raise FileExistsError('No automatic retry/overwrite')
    receipt = dict(status='loading',pid=os.getpid(),gpu=args.gpu,optimizer_steps=0,
        started_at=datetime.now().astimezone().isoformat(),records=[],cache_audits=[],
        scope=__doc__,state_source='step00 generated endpoint/noise interpolants',
        model_forwards=0,clean_forwards=0,
        source_sha256={str(p):sha(p) for p in [Path(__file__),BASE/'current_prefix.py',
            BRIDGE/'geometry_primitives.py',RT/'code/causal/h3_cached.py',MANIFEST]})
    began=time.perf_counter()
    def save():
        receipt['wall_seconds']=time.perf_counter()-began
        tmp=dest.with_suffix('.tmp.json');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(dest)
    save(); caches=[]
    try:
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(RT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval(); checkpoint=BRIDGE/'train_48/step_00'
        trainer=torch.load(checkpoint/'trainer_state.pt',map_location='cpu',weights_only=True)
        assert trainer['optimizer_step']==0
        assert all(sha(checkpoint/n)==h for n,h in trainer['weight_sha256'].items())
        receipt['checkpoint_sha256']=trainer['weight_sha256'];del trainer
        visual=load_adapter(model,checkpoint/'causal_adapter.pt','cuda:0')
        visual_modules=[model.blocks[i].attn.qkv_proj for i in visual['block_indices']]
        configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        bank,_=load_stage1_lora(model,checkpoint/'stage1_lora.pt',device='cuda:0')
        # The zero-update wrappers must be identically zero. Their enabled
        # flags are kept identical for all branches, including Original.
        assert all(not torch.count_nonzero(m.lora_B).item() for m in visual_modules)
        assert all(not torch.count_nonzero(w).item() for m in bank.modules for w in m.lora_B)
        model.requires_grad_(False);versions=[(p,p._version) for p in model.parameters()]
        def move(x):
            if torch.is_tensor(x):return x.to('cuda:0')
            if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
            return x
        prep=json.loads((BRIDGE/'counterfactual/preparation.json').read_text())
        manifest=json.loads(MANIFEST.read_text())
        baseline=json.loads((BRIDGE/'geometry/step_00_step00_generated/probe.json').read_text())
        assert baseline['status']=='complete'
        receipt['status']='running';save()
        with torch.no_grad():
            for clip in dict.fromkeys(r['clip'] for r in prep['records']):
                row=next(r for r in manifest['clips'] if r['clip_id']==clip)
                assert row['split']=='validation' and sha(row['encoded_file'])==row['sha256']
                raw=torch.load(row['encoded_file'],map_location='cpu',weights_only=True)
                generated=BRIDGE/'eval/step_00/generated_30'/clip
                ev=json.loads((generated/'evaluation.json').read_text())
                assert ev['status']=='complete' and ev['encoded_sha256']==row['sha256']
                source=move(torch.load(generated/'latents.pt',map_location='cpu',weights_only=True).float())
                cond=move(raw['conditioning']);noise=cond['initial_noise'].float()
                anchors={0:move(raw['anchors'][0])}
                pipe.load_models_to_device(['video_vae'])
                for k in (1,2):
                    tail=last_frame_image_anchor(pipe.video_vae,source[:,:,:k*5],dtype=pipe.torch_dtype)
                    anchors[k]=torch.cat((cond['anchor'],tail))
                pipe.load_models_to_device(['dit'])
                for chunk in (0,1,2):
                    pr=next(r for r in prep['records'] if r['clip']==clip and r['chunk']==chunk)
                    assert sha(pr['file'])==pr['sha256']
                    pair=move(torch.load(pr['file'],map_location='cpu',weights_only=True))
                    start,stop=pair['start'],pair['stop'];history=source[:,:,:start]
                    def ctx(role):return current_prefix_feedback() if role=='candidate' else nullcontext()
                    def common(k,action,role):
                        return dict(full_packed=pair['packed'],prompt=pair['prompts'][action],anchor=anchors[k],
                            audio=cond['audio_noise'],chunk_frames=5,anchor_frame_index=k*5-1 if k else None,
                            anchor_slot=1,action_prefix_mode='own' if role=='candidate' else 'causal',action_feedback=True)
                    def build(role,action):
                        cache=H3ChunkCache(5,'cpu')
                        with ctx(role):
                            for k in range(chunk):
                                chunk_forward(model,source[:,:,k*5:k*5+5],sigma=0,index=k,
                                    cache=cache,commit=True,**common(k,action,role))
                                receipt['clean_forwards']+=1
                        return cache
                    preds={}; audits=[]
                    for role in ('baseline','candidate'):
                        cache=build(role,'A');caches.append(cache)
                        before=g.cache_digest(cache);cv=g.cache_versions(cache);commits=cache.commits
                        audit=dict(clip=clip,chunk=chunk,role=role,before_sha256=before,
                            cpu_kv_MiB=cache.nbytes/2**20,commits=commits,
                            anchor_sha256={str(k):g.tensor_sha(anchors[k]) for k in range(chunk+1)})
                        if chunk==1:
                            other=build(role,'D')
                            assert g.cache_digest(other)==before
                            audit['current_AD_does_not_change_history_KV']=True;other.clear()
                        with ctx(role):
                            for sigma in SIGMAS:
                                state=(1-sigma)*source[:,:,start:stop]+sigma*noise[:,:,start:stop]
                                ys=[chunk_forward(model,state,sigma=sigma,index=chunk,cache=cache,
                                    **common(chunk,a,role)).float().cpu() for a in 'AD']
                                receipt['model_forwards']+=2
                                preds[role,sigma]=ys
                                if sigma==SIGMAS[1]:
                                    repeat=chunk_forward(model,state,sigma=sigma,index=chunk,cache=cache,
                                        **common(chunk,'A',role)).float().cpu()
                                    receipt['model_forwards']+=1
                                    assert torch.equal(repeat,ys[0]);audit['repeat_RMSE']=0.
                        assert g.cache_digest(cache)==before and cache.commits==commits
                        g.assert_versions(cv);audit.update(after_sha256=before,read_only=True)
                        audits.append(audit);cache.clear();caches.remove(cache)
                    for sigma in SIGMAS:
                        state=(1-sigma)*source[:,:,start:stop]+sigma*noise[:,:,start:stop]
                        ref=[g.forward(model,state,history,cond,pair['prompts'][a],anchors[chunk],
                            pair['packed'],sigma,mode='original').cpu() for a in 'AD']
                        receipt['model_forwards']+=2
                        prior=next(r for r in baseline['records'] if r['clip']==clip and r['chunk']==chunk and r['sigma']==sigma)
                        assert g.tensor_sha(state)==prior['state_sha256'] and g.tensor_sha(history)==prior['history_sha256']
                        record=dict(clip=clip,chunk=chunk,sigma=sigma,state_sha256=g.tensor_sha(state),
                            history_sha256=g.tensor_sha(history),pair_sha256=pr['sha256'],
                            teacher_output_sha256=[g.tensor_sha(x) for x in ref],
                            current_anchor_sha256=g.tensor_sha(anchors[chunk]),roles={})
                        for role in ('baseline','candidate'):
                            ys=preds[role,sigma]
                            assert all(torch.isfinite(x).all() for x in ys+ref)
                            record['roles'][role]=dict(absolute={a:g.compare(ys[i],ref[i]) for i,a in enumerate('AD')},
                                action_delta=g.delta_metrics(ys,ref),prediction_sha256=[g.tensor_sha(x) for x in ys])
                        record['baseline_vs_previous_delta_cos_absdiff']=abs(record['roles']['baseline']['action_delta']['cosine']-prior['action_delta']['cosine'])
                        assert record['baseline_vs_previous_delta_cos_absdiff']<1e-8
                        if chunk==0:
                            assert all(torch.equal(a,b) for a,b in zip(preds['candidate',sigma],ref))
                            record['candidate_first_chunk_original_identity']=True
                        assert all(p._version==v for p,v in versions)
                        receipt['records'].append(record);save()
                        print(json.dumps(dict(clip=clip,chunk=chunk,sigma=sigma,
                            cosine={r:record['roles'][r]['action_delta']['cosine'] for r in ('baseline','candidate')})),flush=True)
                    receipt['cache_audits'].extend(audits);save()
        receipt.update(status='complete',gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:
        for cache in caches:cache.clear()
        save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    if free<34000:raise RuntimeError(f'GPU{args.gpu} needs34000MiB free, got{free}')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2'),str(BRIDGE)]
    main(args)
