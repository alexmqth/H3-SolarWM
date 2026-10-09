"""Fixed generated state, CURRENT-chunk-only A/D intervention; no training.

Main comparison: student r=t versus original-weight instantaneous teacher.
Finite-map output is recorded separately. States are explicitly noised saved
generated endpoints, NOT captured solver states. Teacher has no future video.
"""
from contextlib import contextmanager
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128'
RUNTIME = SOURCE/'training_runtime'
CHECKPOINT = SOURCE/'train_128/step_128'
GENERATED = SOURCE/'eval/anyflow/step_128/generated_8step_native'
SIGMAS = (.9395404663085938, .6894410400390625, .24078089904785155)
TARGETS = (.8694517211914062, .5711835327148438, 0.)


def setup(gpu, reserve=14):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), ABOT_VRAM_RESERVE_GIB=str(reserve),
        HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True',
        ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
        DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
    for key, suffix in [('HF_HOME','hf'), ('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                        ('TRITON_CACHE_DIR','triton'), ('XDG_CACHE_HOME','xdg')]:
        os.environ[key] = str(ROOT/'.cache'/suffix)
    sys.path[:0] = [str(RUNTIME/'code'), str(RUNTIME/'code/abot'),
                   str(RUNTIME/'DiffSynth-Studio-h3-v2')]
    global torch
    import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def tensor_sha(x):
    x = x.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cache_digest(cache):
    h = hashlib.sha256()
    for layer, entries in sorted(cache.layers.items()):
        for e in entries:
            h.update(str((layer,e.index)).encode())
            for x in (e.key,e.value,e.rope): h.update(tensor_sha(x).encode())
    return h.hexdigest()


def cache_versions(cache):
    return [(x,x._version,x.data_ptr()) for es in cache.layers.values()
            for e in es for x in (e.key,e.value,e.rope)]


def assert_versions(items):
    assert all(x._version == v and x.data_ptr() == p for x,v,p in items)


def rms(x):
    return float(x.double().square().mean().sqrt())


def compare(a, b):
    a,b = a.double().flatten(),b.double().flatten()
    an,bn = float(a.norm()),float(b.norm())
    return dict(student_rms=rms(a), teacher_rms=rms(b), student_norm=an,
        teacher_norm=bn, norm_ratio=an/max(bn,1e-30),
        cosine=float(torch.dot(a,b)/(a.norm()*b.norm())) if min(an,bn)>1e-12 else None,
        relative_delta_error=float((a-b).norm())/max(bn,1e-30))


def delta_metrics(student, teacher):
    ds,dt = student[0]-student[1],teacher[0]-teacher[1]
    return dict(**compare(ds,dt), per_latent_frame=[compare(ds[:,:,i],dt[:,:,i])
        for i in range(ds.shape[2])], student_response_relative_to_velocity=
        rms(ds)/max((rms(student[0])+rms(student[1]))/2,1e-30),
        teacher_response_relative_to_velocity=
        rms(dt)/max((rms(teacher[0])+rms(teacher[1]))/2,1e-30))


def replace_current(base, donor, packed, start, stop):
    """Keep head, all past/future action rows, and row layout exactly fixed."""
    prompt = base.clone()
    spans = packed['action_text_spans_local']
    for lo,hi in spans[start:stop]: prompt[int(lo):int(hi)] = donor[int(lo):int(hi)]
    changed = (prompt != base).any(dim=-1)
    allowed = torch.zeros(base.shape[0],dtype=torch.bool,device=base.device)
    for lo,hi in spans[start:stop]: allowed[int(lo):int(hi)] = True
    assert not bool(changed[~allowed].any())
    return prompt


@contextmanager
def original_weights(model, visual, bank, action):
    """Preserve released hotloaded action LoRA; disable only our adaptations."""
    modules = [*visual,*bank.modules] + ([action] if action is not None else [])
    states = [m.enabled for m in modules]
    time_module = model.anyflow_conditioner
    try:
        for m in modules: m.enabled=False
        # model_fn's original path has no target time, including at r=t.
        model.anyflow_conditioner = None
        assert all(not m.enabled for m in modules)
        yield
    finally:
        model.anyflow_conditioner = time_module
        for m,s in zip(modules,states): m.enabled=s


def teacher_forward(model, state, history, cond, prompt, anchor, packed, sigma,
                    *, matched):
    from causal.h3_cached import slice_packed, retime_anchor_position
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    start = history.shape[2]
    whole = torch.cat((history,state),dim=2).to(getattr(model,'_h3_input_dtype',state.dtype))
    rows = (state.shape[-2]//2)*(state.shape[-1]//2)
    local = slice_packed(packed,0,whole.shape[2],rows)
    if matched and start:
        local = retime_anchor_position(local,packed,anchor_rows=anchor.shape[0],
            frame_index=start-1,frame_rows=rows,anchor_slot=1)
    mask = torch.ones_like(whole); mask[:,:,:start] = 0
    return model_fn_minimax_h3(model,whole,cond['audio_noise'].to(whole.dtype),local,prompt,
        timestep_video=torch.tensor(sigma*1000,device=whole.device),
        timestep_audio=torch.tensor(1000.,device=whole.device),
        keyframe_cond_anchor=anchor.to(whole.dtype),input_latents_video=whole,
        denoise_mask_video=mask,fixed_prefix_timesteps=matched)[0][:,:,start:].float()


class RoutingAudit:
    """Inspect the ACTUAL SDPA masks, not a separately reconstructed route."""
    def __init__(self):
        self.records=[]; self.context=None; self.active=False; self.layer=0

    def __enter__(self):
        from causal.h3_cached import ChunkAttention
        self.old_attend=ChunkAttention.attend
        self.old_sdpa=torch.nn.functional.scaled_dot_product_attention
        audit=self
        def attend(control,q,k,v,**kwargs):
            previous=audit.context
            if audit.active and kwargs['layer'] in (0,25,49):
                audit.context=control; audit.layer=kwargs['layer']
            try: return audit.old_attend(control,q,k,v,**kwargs)
            finally: audit.context=previous
        def sdpa(q,k,v,*args,**kwargs):
            if audit.context is not None:
                audit.check(q,k,kwargs['attn_mask'])
            return audit.old_sdpa(q,k,v,*args,**kwargs)
        ChunkAttention.attend=attend
        torch.nn.functional.scaled_dot_product_attention=sdpa
        return self

    def __exit__(self,*exc):
        from causal.h3_cached import ChunkAttention
        ChunkAttention.attend=self.old_attend
        torch.nn.functional.scaled_dot_product_attention=self.old_sdpa

    def check(self,q,k,mask):
        c=self.context; p=c.prefix; fr=c.frame_rows
        h=sum(e.key.shape[0] for e in c.cache.history(self.layer,c.index))
        n=k.shape[-2]-p-h
        spans=c.action_rows.tolist()
        m=mask[0,0]
        row=dict(chunk=c.index,layer=self.layer,history_rows=h,current_rows=n,
                 frame_start=c.frame_start,mask_sha256=tensor_sha(m),
                 history_indices=[e.index for e in c.cache.history(self.layer,c.index)])
        if q.shape[-2] == p:
            assert m.shape == (p,p+h+n)
            assert not bool(m[:,p:p+h].any())
            for frame,(lo,hi) in enumerate(spans):
                expected=torch.arange(n,device=m.device)//fr+c.frame_start==frame
                assert torch.equal(m[lo:hi,p+h:],expected.expand(hi-lo,-1))
            row.update(query='prefix',feedback_own_frame_only=True,history_read=False)
        else:
            assert m.shape == (n,p+h+n)
            assert bool(m[:,p:].all())
            visibility=[]
            for i in range(n//fr):
                frame=c.frame_start+i; flags=[]
                for action,(lo,hi) in enumerate(spans):
                    actual=m[i*fr:(i+1)*fr,lo:hi]
                    assert bool((actual == (action<=frame)).all())
                    flags.append(int(action<=frame))
                visibility.append(flags)
            row.update(query='video',action_visibility=visibility,
                       all_history_and_current_video_visible=True)
        self.records.append(row)


def frozen_inputs():
    files=[OUT/'probe.py', OUT/'check_probe.py', OUT/'cpu_integration.json']
    files += [p for p in CHECKPOINT.glob('*.pt') if p.name!='optimizer.pt']
    files += list((RUNTIME/'code').rglob('*.py'))
    files += list((RUNTIME/'DiffSynth-Studio-h3-v2/diffsynth').rglob('*.py'))
    files += [GENERATED/a/f for a in ('A','D') for f in
              ('cached_latents.pt','conditioning.pt','cached.json','setup.json')]
    return {str(p):sha(p) for p in files}


def main(args):
    destination=OUT/f'probe_{args.history}.json'
    if destination.exists(): raise FileExistsError(destination)
    result=dict(status='preflight',history_action=args.history,pid=os.getpid(),gpu=args.gpu,
        started_at=datetime.now().astimezone().isoformat(),records=[],routing=[],cache_audits=[],
        input_audits=[],checkpoint=str(CHECKPOINT),
        state_protocol='(1-sigma)*saved_generated_current_endpoint + sigma*saved_initial_noise; not actual solver states',
        main_semantics='student r=t vs instantaneous Original H3 noise-clean velocity',
        teacher_scope='generated prefix + current only, no future video; history recomputed bidirectionally',
        teacher_profiles={'matched':'same dual RGB anchors, global anchor positions, fixed prefix times',
                          'native':'single initial anchor, native prefix times'},
        limitations=['single seed/scene; local latent response is not image-space action fidelity',
            'generated endpoints are correlated with the saved noise',
            'teacher prefix retake diagnostic is not original full-horizon inference',
            'student historical hidden states remain cached; teacher recomputes them',
            'causal policy exposes past action rows; native teacher directly exposes own rows only'])
    start_time=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-start_time
        tmp=destination.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(destination)
    save()
    try:
        assert json.loads((OUT/'cpu_integration.json').read_text())['status']=='passed'
        free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        # Reserve14 lowers resident backbone storage relative to the prior
        # reserve6 probe. It changes CPU offload scheduling, not the model.
        if free<36000:
            result.update(status='not_started_gpu_busy',gpu_free_MiB=free);return
        manifest=OUT/'manifest.json'
        if not manifest.exists(): manifest.write_text(json.dumps(frozen_inputs(),indent=2)+'\n')
        for path,digest in json.loads(manifest.read_text()).items():
            assert sha(path)==digest,path
        result.update(manifest_sha256=sha(manifest),gpu_free_at_start_MiB=free,
            gpu_weight_reserve_GiB=14)
        result['status']='loading';save()
        torch.set_num_threads(4);torch.manual_seed(13)
        import infer as abot
        from causal.train_online_selfrollout import action_condition, move_tree
        from causal.pretrained_lora import load_adapter,load_action_residual
        from causal.stage1_lora import load_stage1_lora
        from causal.anyflow import load_anyflow
        from causal.h3_precision import configure_precision,validate_precision_checkpoint
        from causal.h3_cached import (H3ChunkCache,chunk_forward,expand_packed_two_anchors,
                                     last_frame_image_anchor,slice_packed)
        conds={a:move_tree(torch.load(GENERATED/a/'conditioning.pt',map_location='cpu',weights_only=True),'cuda:0')
               for a in ('A','D')}
        cond=conds[args.history]
        for key in ('anchor','audio_noise','initial_noise'):
            assert torch.equal(conds['A'][key],conds['D'][key]),key
        for key,x in conds['A']['packed'].items():
            y=conds['D']['packed'][key]
            assert torch.equal(x,y) if torch.is_tensor(x) else x==y,key
        head=int(cond['packed']['action_text_spans_local'][0][0])
        assert torch.equal(conds['A']['prompt_embeds'][:head],conds['D']['prompt_embeds'][:head])
        generated=torch.load(GENERATED/args.history/'cached_latents.pt',map_location='cuda:0',weights_only=True).float()
        assert generated.shape==(1,24,12,30,52)
        noise=cond['initial_noise'].float()
        packed=expand_packed_two_anchors(cond['packed'],frame_rows=390)
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(
            ROOT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        loaded=load_adapter(model,CHECKPOINT/'causal_adapter.pt','cuda:0')
        visual=[model.blocks[i].attn.qkv_proj for i in loaded['block_indices']]
        action=load_action_residual(model,CHECKPOINT/'action_adapter.pt','cuda:0')['adapter']
        precision=configure_precision(model,'h3_fp32',native_transformer_dir=
            ROOT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        tm,tm_meta=load_anyflow(model,CHECKPOINT/'anyflow_adapter.pt','cuda:0')
        bank,bmeta=load_stage1_lora(model,CHECKPOINT/'stage1_lora.pt',device='cuda:0')
        validate_precision_checkpoint(loaded['metadata'],precision)
        validate_precision_checkpoint(bmeta,precision)
        assert all(m['optimizer_step']==128 for m in (loaded['metadata'],tm_meta,bmeta))
        model.requires_grad_(False);action.requires_grad_(False)
        actions={a:action_condition(a,12,'cuda:0',torch.bfloat16) for a in ('A','D')}
        # All anchors are made from the frozen generated history, before any intervention.
        anchors=[torch.cat((cond['anchor'],cond['anchor']))]
        pipe.load_models_to_device(['video_vae'])
        with torch.no_grad():
            for k in (1,2):
                tail=last_frame_image_anchor(pipe.video_vae,generated[:,:,:k*5],dtype=pipe.torch_dtype)
                anchors.append(torch.cat((cond['anchor'],tail)))
        pipe.load_models_to_device(['dit'])
        versions=[(p,p._version) for p in [*model.parameters(),*action.parameters()]]
        result.update(status='running',precision=precision,
            generated_sha256=tensor_sha(generated),noise_sha256=tensor_sha(noise),
            anchor_sha256=[tensor_sha(a) for a in anchors],bank=bank.describe(),
            shared_base_roles='Original hotloaded H3 action LoRA always retained; our visual/bank/time/residual disabled for teacher')
        cache=H3ChunkCache(5,'cpu'); noised_calls=0; commits=0
        with torch.no_grad(),RoutingAudit() as route:
            for chunk in range(3):
                start=chunk*5;stop=min(start+5,12)
                current=generated[:,:,start:stop];history=generated[:,:,:start]
                common=dict(full_packed=packed,anchor=anchors[chunk],audio=cond['audio_noise'],
                    chunk_frames=5,anchor_frame_index=start-1 if chunk else None,
                    anchor_slot=1 if chunk else 0,action_prefix_mode='causal',
                    action_feedback=True,action_adapter=action)
                if chunk:
                    before=cache_digest(cache); cv=cache_versions(cache);nc=cache.commits
                    prompts={a:replace_current(cond['prompt_embeds'],conds[a]['prompt_embeds'],packed,start,stop)
                             for a in ('A','D')}
                    assert not torch.equal(prompts['A'],prompts['D'])
                    local=slice_packed(packed,start,stop,390)
                    global_rows=packed['img_pos'][780+start*390:780+stop*390]
                    assert torch.equal(local['img_position_ids'][0,local['img_pos'][780:]],
                                       packed['img_position_ids'][0,global_rows])
                    result['input_audits'].append(dict(chunk=chunk,latent_frames=list(range(start,stop)),
                        changed_action_spans=packed['action_text_spans_local'][start:stop],
                        prompt_sha256={a:tensor_sha(p) for a,p in prompts.items()},
                        history_sha256=tensor_sha(history),global_video_positions_match=True,
                        outside_current_action_rows_identical=True,KV_MiB=cache.nbytes/2**20))
                    for index,(sigma,target) in enumerate(zip(SIGMAS,TARGETS)):
                        z=(1-sigma)*current+sigma*noise[:,:,start:stop]
                        zh=tensor_sha(z);record=dict(chunk=chunk,start=start,stop=stop,
                            sigma=sigma,finite_target_sigma=target,z_sha256=zh,
                            history_sha256=tensor_sha(history))
                        tick=time.perf_counter()
                        def student(prompt,act,r):
                            nonlocal noised_calls
                            out=chunk_forward(model,z,index=chunk,cache=cache,sigma=sigma,target_sigma=r,
                                prompt=prompt,action_cond=act,**common).float().cpu()
                            noised_calls+=1;assert_versions(cv)
                            return out
                        route.active=index==0
                        diagonal=[]
                        for a in ('A','D'):
                            diagonal.append(student(prompts[a],actions[a][start:stop],sigma))
                            route.active=False
                        finite=[student(prompts[a],actions[a][start:stop],target) for a in ('A','D')]
                        teachers={}
                        with original_weights(model,visual,bank,action):
                            for profile in ('matched','native'):
                                teachers[profile]=[teacher_forward(model,z,history,cond,prompts[a],
                                    anchors[chunk] if profile=='matched' else cond['anchor'],
                                    packed if profile=='matched' else cond['packed'],sigma,
                                    matched=profile=='matched').cpu() for a in ('A','D')]
                                noised_calls+=2
                            if index==1:
                                repeat=teacher_forward(model,z,history,cond,prompts['A'],anchors[chunk],
                                    packed,sigma,matched=True).cpu();noised_calls+=1
                                record['teacher_repeat_rmse']=rms(repeat-teachers['matched'][0])
                        record['instantaneous']={p:delta_metrics(diagonal,t) for p,t in teachers.items()}
                        # Descriptive ONLY: finite field has different temporal semantics.
                        record['finite_vs_teacher_not_like_for_like']={p:delta_metrics(finite,t) for p,t in teachers.items()}
                        record['finite_vs_diagonal_delta']=compare(finite[0]-finite[1],diagonal[0]-diagonal[1])
                        if index==1:
                            repeat=student(prompts['A'],actions['A'][start:stop],sigma)
                            record['student_repeat_rmse']=rms(repeat-diagonal[0])
                            # 2x2 intervention separates text from the learned residual control input.
                            textA_resD=student(prompts['A'],actions['D'][start:stop],sigma)
                            textD_resA=student(prompts['D'],actions['A'][start:stop],sigma)
                            record['pathway']={
                                'text_effect_residual_A':compare(diagonal[0]-textD_resA,teachers['matched'][0]-teachers['matched'][1]),
                                'text_effect_residual_D':compare(textA_resD-diagonal[1],teachers['matched'][0]-teachers['matched'][1]),
                                'residual_effect_text_A':compare(diagonal[0]-textA_resD,teachers['matched'][0]-teachers['matched'][1]),
                                'residual_effect_text_D':compare(textD_resA-diagonal[1],teachers['matched'][0]-teachers['matched'][1])}
                            if stop<12:
                                opposite='D' if args.history=='A' else 'A'
                                future=replace_current(prompts['A'],conds[opposite]['prompt_embeds'],packed,stop,12)
                                assert not torch.equal(future,prompts['A'])
                                vfuture=student(future,actions['A'][start:stop],sigma)
                                record['future_only_rmse']=rms(vfuture-diagonal[0])
                        assert tensor_sha(z)==zh
                        assert all(torch.isfinite(x).all() for x in [*diagonal,*finite,*teachers['matched'],*teachers['native']])
                        record.update(wall_seconds=time.perf_counter()-tick,
                            output_sha256={p:[tensor_sha(x) for x in xs] for p,xs in
                                dict(diagonal=diagonal,finite=finite,**teachers).items()})
                        result['records'].append(record);result['routing']=route.records
                        result['noisy_forwards']=noised_calls;save()
                        print(json.dumps(dict(chunk=chunk,sigma=sigma,
                            matched=record['instantaneous']['matched']['cosine'],
                            native=record['instantaneous']['native']['cosine'],
                            norm_ratio=record['instantaneous']['matched']['norm_ratio'])),flush=True)
                    assert cache_digest(cache)==before and cache.commits==nc
                    result['cache_audits'].append(dict(chunk=chunk,sha256=before,
                        commits_before_after=nc,unchanged=True,only_video_KV=True))
                    save()
                # Only ORIGINAL baseline actions write history, never counterfactual branches.
                if chunk<2:
                    chunk_forward(model,current,index=chunk,cache=cache,sigma=0.,target_sigma=0.,commit=True,
                        prompt=cond['prompt_embeds'],action_cond=actions[args.history][start:stop],**common)
                    commits+=1
        assert all(p._version==v for p,v in versions)
        result.update(status='complete',noisy_forwards=noised_calls,clean_commit_forwards=commits,
            parameter_versions_unchanged=True,gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            cpu_KV_peak_MiB=cache.peak_bytes/2**20)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally: save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--history',choices=['A','D'],required=True)
    args=ap.parse_args();setup(args.gpu);main(args)
