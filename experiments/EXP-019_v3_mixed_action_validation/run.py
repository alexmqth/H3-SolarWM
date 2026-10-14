"""EXP-019 isolated validation runner; no training and no edits to frozen protocols."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'H3-World/outputs/EXP-019_v3_mixed_action_validation'
FROZEN=ROOT/'H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime'
EXP=ROOT/'submission/experiments'
SCENES=['43866101','7199292c','9dc2e588','b784d995']
FIX=ROOT/'H3-World/outputs/EXP-014_v3_multiscene_teacher/fixtures'
REAL=EXP/'EXP-015_v3_real56_data/artifacts'
VAE=ROOT/'H3-World/outputs/EXP-016_v3_real56_vae'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');q.replace(p)

def setup():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    os.environ.update(ABOT_DIFFSYNTH_ROOT=str(FROZEN/'DiffSynth-Studio-h3-v2'),
        DIFFSYNTH_MODEL_BASE_PATH=str(FROZEN/'DiffSynth-Studio-h3-v2/models'),
        ABOT_VRAM_RESERVE_GIB='18',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    sys.path[:0]=[str(p) for p in [FROZEN/'code',FROZEN/'code/abot',FROZEN/'code/causal',
        FROZEN/'DiffSynth-Studio-h3-v2',EXP/'EXP-002_native_cached',
        ROOT/'submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate',
        EXP/'EXP-017_v3_fixed_action_layout']]
    import torch
    torch.set_num_threads(4)


def fixture(scene):
    import torch
    return torch.load(FIX/f's{SCENES.index(scene)}_{scene}.pt',map_location='cpu',weights_only=True)

def load_tensor(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=True)

def tsha(t):
    import torch
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()

def move(p):
    import torch
    return {k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in p.items()}

def source_paths():
    paths=[Path(__file__),HERE/'taskbook_v1.md',EXP/'EXP-017_v3_fixed_action_layout/fixed_layout.py',
        EXP/'EXP-002_native_cached/interval_cached.py',
        ROOT/'submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py',
        FROZEN/'code/causal/h3_cached.py',FROZEN/'code/causal/local_topology.py',
        FROZEN/'code/causal/h3_precision.py',FROZEN/'code/causal/anyflow_sampling.py',
        FROZEN/'code/abot/infer.py',FROZEN/'checkpoints/H3-World/step-10000.safetensors',
        FROZEN/'DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py',
        FROZEN/'DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py',
        EXP/'EXP-014_v3_multiscene_teacher/source_manifest.json',
        ROOT/'H3-World/outputs/EXP-014_v3_multiscene_teacher/G1/s0_43866101/FM30/first12.pt']
    for i,s in enumerate(SCENES):
        paths += [FIX/f's{i}_{s}.pt',REAL/s/'I0.png',REAL/s/'real56.mp4',REAL/s/'action_script17.json']
        paths += [VAE/s/f'{n}.pt' for n in ['I0','prefix12','full17']]
    return paths

def preflight():
    import torch
    from fixed_layout import frozen_cpu_functions,candidate,action_ids,same_layout,semantic_parts,TOKENIZER_DIR
    from transformers import AutoTokenizer
    import infer as abot
    from interval_cached import interval_cached
    from current_prefix import current_prefix_feedback
    builder,visible=frozen_cpu_functions();tok=AutoTokenizer.from_pretrained(str(TOKENIZER_DIR),local_files_only=True)
    rows=[]
    for s in SCENES:
        f=fixture(s);script=json.loads((REAL/s/'action_script17.json').read_text())
        canon,_=candidate(builder,f['packed'],action_ids(tok,[f['action_sentences']['A']]*37))
        assert same_layout(canon,f['packed'])
        m=json.loads((EXP/f'EXP-016_v3_real56_vae/artifacts/{s}/metrics.json').read_text())
        for name,key in [('I0','I0_latent'),('prefix12','prefix12'),('full17','full17')]:
            assert sha(VAE/s/f'{name}.pt')==m['files'][key]['sha256']
        assert sha(REAL/s/'I0.png')==m['source_I0_sha256']
        assert torch.equal(load_tensor(VAE/s/'prefix12.pt'),load_tensor(VAE/s/'full17.pt')[:,:,:12])
        p,ids=candidate(builder,f['packed'],action_ids(tok,script+[f['action_sentences']['A']]*20))
        parts=[]
        for stop in (12,17):
            v,t=visible(p,ids[:,None],stop,390)
            alt=script+[f['action_sentences']['A']]*20
            alt[stop:]=['A']*(37-stop)
            pp,ii=candidate(builder,f['packed'],action_ids(tok,alt))
            vv,tt=visible(pp,ii[:,None],stop,390)
            assert same_layout(v,vv) and torch.equal(t,tt)
            parts.append(semantic_parts(v,12))
        assert all(torch.equal(parts[0][k],parts[1][k]) for k in parts[0])
        rows.append(dict(scene=s,script=script,unique_sentences=len(set(script+[f['action_sentences']['A']])),
            canonical_exact=True,future_isolation=True,vae_prefix_exact=True))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
    write(HERE/'SOURCE_MANIFEST.json',manifest)
    write(HERE/'CPU_PREFLIGHT.json',dict(status='PASS',gpu_calls=0,scenes=rows,
        text_encoder_upper_count=sum(1+r['unique_sentences'] for r in rows),
        runtime=str(FROZEN),canonical_history='EXP-014 s0 FM30 first12; rebuild both caches',
        scope='CPU only; actual model/mixed embedding not yet tested'))
    print('P0 PASS',flush=True)

class Ledger:
    def __init__(self,stage,gpu):
        self.stage=stage;self.t=time.perf_counter();self.dir=OUT/stage
        if self.dir.exists():raise FileExistsError(self.dir)
        marker=json.loads((HERE/f'judge/{stage}_APPROVED.json').read_text())
        assert marker['source_manifest_sha256']==sha(HERE/'SOURCE_MANIFEST.json')
        for p,h in json.loads((HERE/'SOURCE_MANIFEST.json').read_text()).items():
            if sha(ROOT/p)!=h:raise RuntimeError(f'source changed: {p}')
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu)
        info=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=name,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip()
        assert int(info.split(',')[-1])>=44000,info
        assert shutil.disk_usage(OUT).free>40*2**30
        self.dir.mkdir();self.counts={'forward':0,'commit':0,'decode':0,'text':0}
        self.limits={'forward':10 if stage=='R' else (0 if stage=='E' else 39),
            'commit':2 if stage=='R' else (0 if stage=='E' else 1),
            'decode':0 if stage in ('R','E') else 2,'text':14 if stage=='E' else 0}
        self.d=dict(task='EXP-019/v1',stage=stage,status='running',gpu=gpu,gpu_info=info,
            counts=self.counts,limits=self.limits,started=time.time(),elapsed=0.,events=[])
        self.save()
    def save(self):
        self.d['elapsed']=time.perf_counter()-self.t;write(self.dir/'budget.json',self.d)
    def reserve(self,key):
        if time.perf_counter()-self.t>=900:raise TimeoutError('stage 900s cap')
        if self.counts[key]>=self.limits[key]:raise RuntimeError(f'cap {key}')
        self.counts[key]+=1;self.d['events'].append(dict(kind=key,count=self.counts[key],time=time.time()));self.save()
    def peak(self):
        import torch
        p=torch.cuda.max_memory_allocated()/2**30
        if p>44:raise MemoryError(f'allocated peak {p}')
        return p
    def close(self,status,error=None):
        import torch
        self.d.update(status=status,error=error,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        self.save()

def model():
    import torch,infer as abot
    from causal.h3_precision import configure_precision
    tick=time.perf_counter();pipe=abot.load_pipeline('cuda:0')
    pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(FROZEN/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
    m=pipe.dit.eval().requires_grad_(False)
    precision=configure_precision(m,'h3_fp32',native_transformer_dir=FROZEN/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
    pipe.load_models_to_device(['dit']);torch.cuda.synchronize()
    return pipe,m,dict(precision=precision,load_seconds=time.perf_counter()-tick,
        torch=torch.__version__,cuda=torch.version.cuda,attention_backend='torch SDPA via frozen current_prefix attention',reserve_gib=18)

def forward(m,x,f,cache,index,sigma,ledger,commit=False):
    import torch
    from causal.local_topology import visible_inputs
    from interval_cached import interval_cached
    from current_prefix import current_prefix_feedback
    stop=12 if index==0 else 17
    p,pr=visible_inputs(move(f['packed']),f['prompt'].to('cuda:0'),stop,390)
    ledger.reserve('forward')
    if commit:ledger.reserve('commit')
    with torch.no_grad(),current_prefix_feedback():
        y=interval_cached(m,x.to('cuda:0'),start=0 if index==0 else 12,index=index,cache=cache,
            full_packed=p,prompt=pr,anchor=f['anchor'].to('cuda:0'),audio=f['audio_noise'].to('cuda:0'),sigma=sigma,commit=commit)
    if not torch.isfinite(y).all():raise FloatingPointError('velocity nonfinite')
    ledger.peak();return y

def diff(a,b):
    import torch
    d=a.float()-b.float()
    return dict(exact=torch.equal(a,b),max_abs=float(d.abs().max()),
        relative_rms=float(d.square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-12)))

def regression(l):
    import torch
    from causal.h3_cached import H3ChunkCache
    from fixed_layout import frozen_cpu_functions,candidate,action_ids,TOKENIZER_DIR,same_layout
    from transformers import AutoTokenizer
    builder,_=frozen_cpu_functions();tok=AutoTokenizer.from_pretrained(str(TOKENIZER_DIR),local_files_only=True)
    f=fixture(SCENES[0]);f['prompt']=f['prompts']['A']
    p,_=candidate(builder,f['packed'],action_ids(tok,[f['action_sentences']['A']]*37))
    assert same_layout(p,f['packed']);new=dict(f,packed=p)
    hist=load_tensor(ROOT/'H3-World/outputs/EXP-014_v3_multiscene_teacher/G1/s0_43866101/FM30/first12.pt')
    caches=[H3ChunkCache(max_history=5,storage_device='cpu') for _ in range(2)]
    pipe,m,meta=model();rows=[]
    for idx in (0,1):
        if idx:
            for ff,c in zip((f,new),caches):
                delout=forward(m,hist,ff,c,0,0.,l,True);del delout
            cache_rows=[]
            for layer in caches[0].layers:
                for a,b in zip(caches[0].layers[layer],caches[1].layers[layer]):
                    assert a.index==b.index==0
                    for name in ('key','value','rope'):
                        row=diff(getattr(a,name),getattr(b,name));cache_rows.append(dict(layer=layer,tensor=name,**row))
            write(l.dir/'cache_comparison.json',cache_rows)
            assert len(caches[0].layers)==len(caches[1].layers)==50
            assert all(r['exact'] for r in cache_rows)
        x=f['initial_noise'][:,:,:12] if idx==0 else f['initial_noise'][:,:,12:17]
        for sigma in (1.,.5):
            state=x.float()*sigma
            a=forward(m,state,f,caches[0],idx,sigma,l).cpu()
            b=forward(m,state,new,caches[1],idx,sigma,l).cpu()
            row=dict(index=idx,sigma=sigma,state_sha256=tsha(state),**diff(a,b));rows.append(row)
            write(l.dir/'comparisons.json',rows)
            print('REGRESSION',row,flush=True)
            assert row['exact'],'canonical numerical mismatch'
    write(l.dir/'result.json',dict(status='PASS',metadata=meta,comparisons=rows,cache_tensors_exact=True,
        counts=l.counts,peak_allocated_gib=l.peak(),elapsed=time.perf_counter()-l.t,
        state_note='sigma1 = frozen initial noise; sigma.5 = same noise scaled by .5, fixed probe not ODE trajectory'))


def encode(l):
    import torch
    from PIL import Image
    from fixed_layout import frozen_cpu_functions,candidate,action_ids,same_layout,semantic_parts
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline,ModelConfig,MiniMaxH3Unit_PromptEmbedder,patchify_video
    from causal.local_topology import visible_inputs
    dtype=torch.bfloat16
    cfg=dict(offload_dtype=dtype,offload_device='cpu',onload_dtype=dtype,onload_device='cpu',
        preparing_dtype=dtype,preparing_device='cuda:0',computation_dtype=dtype,computation_device='cuda:0')
    pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=dtype,device='cuda:0',model_configs=[
        ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern='FL2VA/text_encoder/model*.safetensors',**cfg)],
        processor_config=ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern='FL2VA/processor/'),vram_limit=27.)
    hook=pipe.text_encoder.register_forward_pre_hook(lambda *a:l.reserve('text'))
    builder,_=frozen_cpu_functions();unit=MiniMaxH3Unit_PromptEmbedder();rows=[]
    with torch.no_grad():
        for s in SCENES:
            old=fixture(s);script=json.loads((REAL/s/'action_script17.json').read_text())
            lines=script+[old['action_sentences']['A']]*20
            result=unit.process(pipe,old['scene_static'],keyframes=[Image.open(REAL/s/'I0.png').convert('RGB')],
                height=480,width=832,action_script=lines)
            prompt=result['prompt_embeds'].cpu();spans=result['action_text_spans']
            assert spans[0][0]==old['packed']['action_text_spans_local'][0][0]
            p,ids=candidate(builder,old['packed'],action_ids(pipe.tokenizer,lines))
            assert spans==p['action_text_spans_local']
            assert torch.equal(result['text_token_tags'].cpu(),p['token_tags'][p['text_pos']])
            assert len(prompt)==len(p['text_pos']) and torch.isfinite(prompt).all()
            clean=patchify_video(load_tensor(VAE/s/'I0.pt')).to('cuda:0',dtype)
            noise=pipe.generate_noise((1,24,38,30,52),13,'cpu',dtype)[:,:,:1]
            nr=patchify_video(noise).to('cuda:0',dtype);ts=torch.tensor(.999,device='cuda:0',dtype=dtype)
            anchor=(ts*clean+(1-ts)*nr).contiguous().cpu()
            checks=[];sem=[]
            for stop in (12,17):
                vis,vpr=visible_inputs(p,prompt,stop,390)
                alt_ids=action_ids(pipe.tokenizer,lines[:stop]+['A']*(37-stop))
                alt,_=candidate(builder,old['packed'],alt_ids)
                head=spans[0][0]
                # Different future length and arbitrary embedding values; cropped real rows must be exact.
                prefix_end=spans[stop-1][1]
                alt_prompt=torch.cat([prompt[:prefix_end],torch.ones((len(alt['text_pos'])-prefix_end,prompt.shape[1]),dtype=prompt.dtype)])
                av,ap=visible_inputs(alt,alt_prompt,stop,390)
                assert same_layout(vis,av) and torch.equal(vpr,ap)
                sem.append(semantic_parts(vis,12));checks.append(dict(stop=stop,visible_exact=True))
            assert all(torch.equal(sem[0][k],sem[1][k]) for k in sem[0])
            item=dict(format='exp019_mixed_native_v1',scene=s,packed=p,prompt=prompt,anchor=anchor,
                initial_noise=old['initial_noise'],audio_noise=old['audio_noise'],script=script,
                source_I0_sha256=sha(REAL/s/'I0.png'))
            path=l.dir/f'{s}.pt';torch.save(item,path)
            rows.append(dict(scene=s,path=str(path.relative_to(ROOT)),sha256=sha(path),checks=checks,
                head_rows=head,text_rows=len(prompt),token_lengths=[hi-lo for lo,hi in spans[:17]],
                head_reencoded=True,semantic_history_stable=True,anchor_sha256=tsha(anchor)))
            write(l.dir/'result.json',dict(status='running',fixtures=rows,counts=l.counts));l.peak()
            print('ENCODED',s,flush=True)
    hook.remove();assert l.counts['text']==14,l.counts
    write(l.dir/'result.json',dict(status='PASS',fixtures=rows,counts=l.counts,elapsed=time.perf_counter()-l.t))


def cache_hash(cache):
    return {str(k):[{'index':e.index,'key':tsha(e.key),'value':tsha(e.value),'rope':tsha(e.rope)} for e in v]
        for k,v in cache.layers.items()}

def read_video(path):
    import numpy as np
    raw=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'])
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,480,832,3).copy()

def save_video(frames,path):
    import numpy as np
    h,w=frames.shape[1:3]
    p=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','24',
        '-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-y',str(path)],stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    if p.returncode:raise RuntimeError('video encode failure')

def sheet(frames,path,start=0):
    from PIL import Image,ImageDraw
    n=len(frames);im=Image.new('RGB',(4*416,((n+3)//4)*264),'black');draw=ImageDraw.Draw(im)
    for i,f in enumerate(frames):
        x=(i%4)*416;y=(i//4)*264;im.paste(Image.fromarray(f).resize((416,240)),(x,y));draw.text((x+5,y+243),f'RGB {start+i}',fill='white')
    im.save(path,quality=92)

def generation(l,scene):
    import torch,numpy as np
    from causal.h3_cached import H3ChunkCache
    from causal.anyflow_sampling import configure_video_schedule
    er=json.loads((OUT/'E/result.json').read_text());entry=next(r for r in er['fixtures'] if r['scene']==scene)
    assert er['status']=='PASS' and sha(ROOT/entry['path'])==entry['sha256']
    f=load_tensor(ROOT/entry['path']);hist=load_tensor(VAE/scene/'prefix12.pt');hist_sha=tsha(hist)
    source=read_video(REAL/scene/'real56.mp4');recon=read_video(VAE/scene/'reconstruction.mp4')
    assert source.shape==recon.shape==(56,480,832,3)
    pipe,m,meta=model();cache=H3ChunkCache(max_history=5,storage_device='cpu')
    tick=time.perf_counter();y=forward(m,hist,f,cache,0,0.,l,True);del y;torch.cuda.synchronize()
    commit_seconds=time.perf_counter()-tick;ch=cache_hash(cache)
    assert len(cache.layers)==50 and all([e.index for e in v]==[0] for v in cache.layers.values())
    kv_bytes=sum(t.numel()*t.element_size() for v in cache.layers.values() for e in v for t in (e.key,e.value,e.rope))
    write(l.dir/'cache_before.json',ch)
    runs=[];videos={}
    for steps in (30,8):
        pipe.load_models_to_device(['dit']);sigmas=configure_video_schedule(pipe.scheduler,steps=steps,grid='native',flow_shift=2.22)
        current=f['initial_noise'][:,:,12:17].to('cuda:0',torch.float32).clone()
        initial=tsha(current);torch.cuda.synchronize();tick=time.perf_counter()
        with torch.no_grad():
            for i,t in enumerate(pipe.scheduler.timesteps):
                y=forward(m,current,f,cache,1,float(t)/1000,l)
                current=pipe.scheduler.step(y,t,current)
                if not torch.isfinite(current).all():raise FloatingPointError('sample nonfinite')
                print(f'{scene} FM{steps} {i+1}/{steps}',flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        assert cache_hash(cache)==ch and tsha(hist)==hist_sha
        torch.save(current.cpu(),l.dir/f'fm{steps}_C2.pt')
        pipe.load_models_to_device(['video_vae']);l.reserve('decode');torch.cuda.synchronize();tick=time.perf_counter()
        with torch.no_grad():
            lat=torch.cat([hist.to('cuda:0'),current],dim=2)
            rgb=pipe.video_vae.decode_video(lat,dtype=pipe.torch_dtype,process_image=False,tiled=True,tile_size=256,tile_overlap=64)
            frames=np.stack([np.asarray(z.convert('RGB'),dtype=np.uint8) for z in pipe.vae_output_to_video(rgb,min_value=0,max_value=1)])
        torch.cuda.synchronize();decode_seconds=time.perf_counter()-tick
        assert frames.shape==(56,480,832,3)
        save_video(frames,l.dir/f'fm{steps}_raw_decode56.mp4')
        display=frames.copy();display[:39]=source[:39]
        save_video(display,l.dir/f'fm{steps}_gt39_plus_generated17.mp4');sheet(display[35:],l.dir/f'fm{steps}_boundary_and_new.jpg',35)
        videos[steps]=display
        row=dict(nfe_per_chunk=steps,new_rgb_frames=17,display_frames=56,history_kind='GT C1',
            initial_noise_sha256=initial,sigmas=sigmas,sampling_seconds=seconds,decode_seconds=decode_seconds,
            endpoint_sha256=sha(l.dir/f'fm{steps}_C2.pt'),history_tensor_unchanged=True,history_kv_unchanged=True,
            displayed_prefix_exact_to_decoded_source=True,peak_allocated_gib=l.peak(),
            boundary_rgb_MAD=float(np.abs(display[39].astype(float)-display[38]).mean()),
            boundary_raw_decoder_MAD=float(np.abs(frames[39].astype(float)-frames[38]).mean()))
        runs.append(row);write(l.dir/'result.json',dict(status='running',runs=runs))
        del current,y,rgb,lat;gc.collect()
    # All panels explicitly labelled; only last 17 frames of FM panels are generated.
    from PIL import Image,ImageDraw
    combined=[]
    for j in range(56):
        canvas=Image.new('RGB',(832*3,512),'black');d=ImageDraw.Draw(canvas)
        for col,(label,arr) in enumerate([('SOURCE / GT',source),('FM-30NFE | GT prefix' if j<39 else 'FM-30NFE | GENERATED',videos[30]),('FM-8NFE | GT prefix' if j<39 else 'FM-8NFE | GENERATED',videos[8])]):
            canvas.paste(Image.fromarray(arr[j]),(col*832,32));d.text((col*832+8,8),label+f' | frame {j}',fill='white')
        combined.append(np.asarray(canvas))
    save_video(np.stack(combined),l.dir/'source_vs_fm30_vs_fm8.mp4')
    write(l.dir/'result.json',dict(status='COMPLETE_PENDING_JUDGE',scene=scene,metadata=meta,runs=runs,
        script=f['script'],commit_seconds=commit_seconds,cpu_kv_bytes=kv_bytes,counts=l.counts,
        elapsed=time.perf_counter()-l.t,peak_allocated_gib=l.peak(),
        limitations=['GT history continuation; not free running','only frames39..55 newly generated',
            '4 train episodes, no independent generalization claim','no D counterfactual GT']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['P0','R','E']+SCENES);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    setup()
    if a.stage=='P0':preflight();return
    l=Ledger(a.stage,a.gpu)
    import torch
    torch.manual_seed(13);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    try:
        if a.stage=='R':regression(l)
        elif a.stage=='E':encode(l)
        else:generation(l,a.stage)
        l.close('complete')
    except BaseException as e:
        l.close('failed',repr(e));raise
if __name__=='__main__':main()
