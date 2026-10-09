"""Current-chunk A/D intervention on held-out real/causal generated states.

Instantaneous FM predictions. Each checkpoint rebuilds its own history KV;
teacher uses original weights, full prefix and matched current RGB anchors.
Teacher history is bidirectionally recomputed, student history stays cached.
"""
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
RT=BASE/'runtime'
MANIFEST=BASE.parents[2]/'data/abot_bridge/encoded_manifest.json'
CLIPS=['118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750']
SIGMAS=(.9395404663085938,.6894410400390625,.24078089904785155)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(args):
    import torch
    import infer as abot
    import geometry_primitives as g
    from causal.h3_cached import H3ChunkCache,chunk_forward,last_frame_image_anchor
    from causal.h3_precision import configure_precision,validate_precision_checkpoint
    from causal.pretrained_lora import load_adapter
    from causal.stage1_lora import load_stage1_lora
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    from causal.h3_cached import slice_packed,retime_anchor_position
    torch.set_num_threads(4);torch.manual_seed(13)
    checkpoint=BASE/f'train_48/step_{args.step:02d}'
    run=BASE/'geometry'/f'step_{args.step:02d}_{args.history}'
    run.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='preflight',pid=os.getpid(),gpu=args.gpu,step=args.step,history_source=args.history,
        started_at=datetime.now().astimezone().isoformat(),records=[],cache_audits=[],
        scope='same-state instantaneous current-only A/D; persistent CPU KV student vs original full-prefix teacher',
        state_construction='fixed endpoint-noise interpolants, not captured solver states',
        teacher_backend='SDPA original directed predicate, matched backend with causal',
        limitation='Teacher recomputes bidirectional history; cached student does not. Counterfactual action layout may differ from natural recorded layout; all KV rebuilt.',
        model_forwards=0,clean_forwards=0)
    tick=time.perf_counter();cache=None
    def save():
        receipt['wall_seconds']=time.perf_counter()-tick
        tmp=run/'probe.tmp.json';tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(run/'probe.json')
    save()
    try:
        prep=json.loads((BASE/'counterfactual/preparation.json').read_text());assert prep['status']=='complete'
        train=torch.load(checkpoint/'trainer_state.pt',map_location='cpu',weights_only=True)
        assert train['optimizer_step']==args.step and train['config']['objective']=='fm'
        assert train['config']['real_manifest_sha256']==sha(MANIFEST)
        assert all(sha(checkpoint/n)==h for n,h in train['weight_sha256'].items())
        receipt['checkpoint_sha256']=train['weight_sha256'];del train
        receipt['source_sha256']={n:sha(BASE/n) for n in ('probe_real_geometry.py','geometry_primitives.py','prepare_counterfactual.py')}
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(RT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        loaded=load_adapter(model,checkpoint/'causal_adapter.pt','cuda:0')
        visual=[model.blocks[i].attn.qkv_proj for i in loaded['block_indices']]
        precision=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        bank,meta=load_stage1_lora(model,checkpoint/'stage1_lora.pt',device='cuda:0')
        validate_precision_checkpoint(loaded['metadata'],precision);validate_precision_checkpoint(meta,precision)
        model.requires_grad_(False);versions=[(p,p._version) for p in model.parameters()]
        def teacher_role(enabled):
            for m in [*visual,*bank.modules]:m.enabled=not enabled
        def move(v):
            if torch.is_tensor(v):return v.to('cuda:0')
            if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
            return v
        manifest=json.loads(MANIFEST.read_text());receipt['status']='running';save()
        with torch.no_grad():
            for clip in CLIPS:
                row=next(r for r in manifest['clips'] if r['clip_id']==clip)
                assert row['split']=='validation' and sha(row['encoded_file'])==row['sha256']
                raw=torch.load(row['encoded_file'],map_location='cpu',weights_only=True)
                source=raw['clean_latents'].float()
                generated_path=None
                if args.history!='gt':
                    source_step=0 if args.history=='step00_generated' else args.step
                    directory=BASE/f'eval/step_{source_step:02d}/generated_30'/clip
                    evaluation=json.loads((directory/'evaluation.json').read_text())
                    assert evaluation['status']=='complete' and evaluation['encoded_sha256']==row['sha256']
                    generated_path=directory/'latents.pt';source=torch.load(generated_path,map_location='cpu',weights_only=True).float()
                source=source.to('cuda:0');cond=move(raw['conditioning']);noise=cond['initial_noise'].float()
                assert source.shape==noise.shape and torch.isfinite(source).all()
                anchors=move({i:x for i,x in enumerate(raw['anchors'])})
                if args.history!='gt':
                    pipe.load_models_to_device(['video_vae'])
                    for k in (1,2):
                        tail=last_frame_image_anchor(pipe.video_vae,source[:,:,:k*5],dtype=pipe.torch_dtype)
                        anchors[k]=torch.cat((cond['anchor'],tail))
                pipe.load_models_to_device(['dit'])
                for chunk in (0,1,2):
                    pair_row=next(r for r in prep['records'] if r['clip']==clip and r['chunk']==chunk)
                    assert sha(pair_row['file'])==pair_row['sha256'] and pair_row['source_sha256']==row['sha256']
                    pair=move(torch.load(pair_row['file'],map_location='cpu',weights_only=True))
                    packed=pair['packed'];start=pair['start'];stop=pair['stop'];history=source[:,:,:start]
                    def common(k,action):
                        return dict(full_packed=packed,prompt=pair['prompts'][action],anchor=anchors[k],audio=cond['audio_noise'],
                            chunk_frames=5,anchor_frame_index=k*5-1 if k else None,anchor_slot=1,
                            action_prefix_mode='causal',action_feedback=True,action_adapter=None)
                    def build(action):
                        c=H3ChunkCache(5,'cpu')
                        for k in range(chunk):
                            chunk_forward(model,source[:,:,k*5:k*5+5],index=k,cache=c,sigma=0.,commit=True,**common(k,action))
                            receipt['clean_forwards']+=1
                        return c
                    teacher_role(False);cache=build('A');before=g.cache_digest(cache);cv=g.cache_versions(cache);commits=cache.commits
                    audit=dict(clip=clip,chunk=chunk,before_sha256=before,commits=commits,cpu_kv_MiB=cache.nbytes/2**20,
                        rebuilt_from_raw_history=True,shared_A_D_cache=True)
                    # Current A/D is future to every historical chunk. Verify
                    # real full K/V equality once, beyond a mask-only check.
                    if chunk==1:
                        other=build('D');audit['rebuilt_D_history_equals_A']=g.cache_digest(other)==before
                        assert audit['rebuilt_D_history_equals_A'];other.clear()
                    for sigma in SIGMAS:
                        z=(1-sigma)*source[:,:,start:stop]+sigma*noise[:,:,start:stop]
                        zhash=g.tensor_sha(z);teacher_role(False)
                        student=[chunk_forward(model,z,index=chunk,cache=cache,sigma=sigma,**common(chunk,a)).float().cpu() for a in 'AD']
                        receipt['model_forwards']+=2
                        teacher_role(True)
                        teacher=[g.forward(model,z,history,cond,pair['prompts'][a],anchors[chunk],packed,sigma,mode='original').cpu() for a in 'AD']
                        receipt['model_forwards']+=2
                        record=dict(clip=clip,chunk=chunk,sigma=sigma,state_sha256=zhash,history_sha256=g.tensor_sha(history),
                            pair_sha256=pair_row['sha256'],source_sha256=row['sha256'],
                            endpoint_source='real GT' if generated_path is None else str(generated_path),
                            endpoint_sha256=g.tensor_sha(source),
                            absolute={a:g.compare(student[i],teacher[i]) for i,a in enumerate('AD')},
                            action_delta=g.delta_metrics(student,teacher))
                        if sigma==SIGMAS[1]:
                            repeated=g.forward(model,z,history,cond,pair['prompts']['A'],anchors[chunk],packed,sigma,mode='original').cpu()
                            receipt['model_forwards']+=1;record['teacher_repeat_rmse']=g.rms(repeated-teacher[0])
                            teacher_role(False)
                            repeated=chunk_forward(model,z,index=chunk,cache=cache,sigma=sigma,**common(chunk,'A')).float().cpu()
                            receipt['model_forwards']+=1;record['student_repeat_rmse']=g.rms(repeated-student[0])
                            assert record['teacher_repeat_rmse']==record['student_repeat_rmse']==0
                        assert all(torch.isfinite(x).all() for x in student+teacher)
                        assert all(p._version==v for p,v in versions)
                        assert g.tensor_sha(z)==zhash;g.assert_versions(cv)
                        receipt['records'].append(record);save()
                        print(json.dumps(dict(clip=clip,chunk=chunk,sigma=sigma,cosine=record['action_delta']['cosine'])),flush=True)
                    assert g.cache_digest(cache)==before and cache.commits==commits
                    audit.update(after_sha256=before,read_only=True)
                    receipt['cache_audits'].append(audit);cache.clear();cache=None;save()
        receipt.update(status='complete',gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:receipt.update(status='failed',error=repr(exc));raise
    finally:
        if cache is not None:cache.clear()
        save()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--step',type=int,choices=[0,48],required=True)
    ap.add_argument('--history',choices=['gt','step00_generated','own_generated'],required=True)
    args=ap.parse_args()
    if args.step==0 and args.history=='own_generated':ap.error('Use step00_generated for zero checkpoint')
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<34000:raise RuntimeError(f'GPU{args.gpu} needs34000MiB free, got{free}')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
