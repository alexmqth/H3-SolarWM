"""Preselected step4 videos: held-out GT or same-state parking A/D forks."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'


def main(args):
    import av
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import torch
    import infer as abot
    from causal.benchmark import write_video
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.local_transition import transition_forward
    from causal.anyflow_sampling import configure_video_schedule
    from causal.evaluate_action_control import evaluate as flow_metrics
    from eval_common import sha,tensor_hash as th,move,load_bank,terminal
    torch.set_num_threads(4);torch.manual_seed(13)
    spec=json.loads((BASE/'evaluation_protocol.json').read_text())
    for name,want in spec['sources'].items():assert sha(BASE/name)==want,name
    for rel,want in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==want,rel
    training=json.loads((BASE/f'train_{args.arm}/training.json').read_text());terminal(training)
    assert training['status']=='complete_pending_evaluation' and training['optimizer_updates']==4
    for step in (0,4):assert sha(BASE/f'train_{args.arm}/step_{step:02d}/action_lora.pt')==spec['checkpoints'][args.arm][str(step)]
    out=BASE/f'eval_{args.arm}_{args.suite}';out.mkdir(exist_ok=False);begin=time.perf_counter()
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=args.gpu,
        arm=args.arm,suite=args.suite,protocol_sha256=sha(BASE/'evaluation_protocol.json'),records=[],
        sampling_forwards=0,diagnostic_forwards=0,optimizer_updates=0,CPU_hidden_KV_MiB=0,
        timing_scope='Recorded local sampling; concurrent shared-host jobs; not speedup or warmup-averaged benchmark')
    def save():
        result['wall_seconds']=time.perf_counter()-begin
        p=out/'evaluation.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'evaluation.json')
    def decoded(path):
        with av.open(str(path)) as c:return [f.to_image().convert('RGB') for f in c.decode(video=0)]
    save()
    try:
        cases=[]
        if args.suite=='gt':
            enc=BASE/'encoded_validation/encoding.json';assert sha(enc)==spec['GT_encoding_sha256']
            baseline=BASE/'gt_history_baseline/evaluation.json';assert sha(baseline)==spec['GT_baseline_sha256']
            originals=json.loads(baseline.read_text())
            for rec in json.loads(enc.read_text())['clips']:
                assert sha(rec['encoded_path'])==rec['sha256']
                d=torch.load(rec['encoded_path'],map_location='cpu',weights_only=True)
                source=next(r for r in originals['records'] if r['clip_id']==rec['clip_id'])
                zero_path=Path(source['video']).parent/'current.mp4'
                assert sha(source['video'])==source['video_sha256']
                with np.load(d['source']['path'],allow_pickle=False) as raw:
                    gt=[Image.fromarray(x) for x in raw['rgb'][81:120]]
                    context=[Image.fromarray(x) for x in raw['rgb'][73:81]]
                cases.append(dict(name=rec['clip_id'],history=d['safe_prefixes'][24],noise=d['initial_noise'],
                    anchor=d['anchor'],audio=d['audio_noise'],packed=d['positive']['packed'],prompt=d['positive']['prompt_embeds'],
                    start=24,stop=36,index=2,r0=81,r1=120,history_kind='real_GT',zero_video=zero_path,
                    expected_input_sha256=rec['sha256'],source=source,gt=gt,context=context))
            expected_field=None
        else:
            ref=args.suite[-1];old=Path(spec['parking_root']);coarse=old/'source_coarse'
            inp=coarse/f'inputs/parking_{ref}.pt';assert sha(inp)==spec['parking_inputs'][ref]
            d=torch.load(inp,map_location='cpu',weights_only=True)
            original_path=old/f'window1_{ref}/evaluation.json';assert sha(original_path)==spec['parking_baselines'][ref]
            originals=json.loads(original_path.read_text())
            for action in 'AD':
                source=next(r for r in originals['records'] if r['action']==action)
                video=old/f'window1_{ref}/{action}.mp4';assert sha(video)==source['video_sha256']
                cases.append(dict(name=f'parking_history{ref}_current{action}',history=d['reference_latents'][:,:,:12],
                    noise=d['initial_noise'],anchor=d['anchor'],audio=d['audio_noise'],packed=d['packed'],prompt=d['pairs'][1]['prompts'][action],
                    start=12,stop=24,index=1,r0=39,r1=81,history_kind='Original_generated_fixed_reference_not_GT',
                    zero_video=video,source=source,original_receipt=originals,action=action,expected_input_sha256=sha(inp),gt=None,context=None))
            fields_path=old/f'probe_{ref}/probe_fields.pt';assert sha(fields_path)==spec['parking_probe_fields'][ref]
            fields=torch.load(fields_path,map_location='cpu',weights_only=True)
            expected_field=next(r for r in fields if r['window']==1 and r['step']==0)['NA'];del fields
        pipe=abot.load_pipeline('cuda:0');released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        precision=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        result['precision']=precision;pipe.load_models_to_device(['dit'])
        sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22);result['sigmas']=sigmas
        c=cases[0];layout,prompt=visible_inputs(move(c['packed']),c['prompt'].to('cuda:0'),c['stop'],390)
        state=c['noise'][:,:,c['start']:c['stop']].to('cuda:0',torch.float32)
        kw=dict(history=c['history'].to('cuda:0',torch.float32),history_noise=c['noise'][:,:,:c['start']].to('cuda:0',torch.float32),
                full_packed=layout,prompt=prompt,anchor=c['anchor'].to('cuda:0'),audio=c['audio'].to('cuda:0'),sigma=1.,index=c['index'])
        with torch.no_grad():
            before=transition_forward(model,state,**kw);result['diagnostic_forwards']+=1
            if expected_field is not None:
                result['archived_N_field_replay_max_abs']=float((before.cpu()-expected_field).abs().max())
                assert torch.equal(before.cpu(),expected_field)
            _,zero_meta=load_bank(model,BASE/f'train_{args.arm}/step_00/action_lora.pt',precision=precision,check_original=True)
            after=transition_forward(model,state,**kw);result['diagnostic_forwards']+=1
            result['zero_checkpoint_identity_max_abs']=float((after-before).abs().max());assert torch.equal(after,before)
            adapter,metadata=load_bank(model,BASE/f'train_{args.arm}/step_04/action_lora.pt',precision=precision)
            assert metadata['optimizer_step']==4 and metadata['arm']==args.arm
            versions=[(p,p._version) for p in list(model.parameters())+adapter]
            del before,after,state,kw,expected_field
            torch.cuda.reset_peak_memory_stats();save()
            for c in cases:
                folder=out/c['name'];folder.mkdir()
                history=c['history'].to('cuda:0',torch.float32);noise=c['noise'].to('cuda:0',torch.float32)
                hh,nh=th(history),th(noise)
                layout,prompt=visible_inputs(move(c['packed']),c['prompt'].to('cuda:0'),c['stop'],390)
                anchor=c['anchor'].to('cuda:0');audio=c['audio'].to('cuda:0')
                if args.suite!='gt':
                    oldr=c['original_receipt'];assert sigmas==oldr['sigmas']
                    assert hh==oldr['history_sha256'] and th(noise[:,:,:c['start']])==oldr['history_noise_sha256']
                    assert th(noise[:,:,c['start']:c['stop']])==oldr['current_initial_noise_sha256']
                    assert th(anchor)==oldr['anchor_sha256'] and th(audio)==oldr['audio_sha256']
                    assert th(prompt)==c['source']['prompt_sha256'] and th(layout['img_position_ids'])==c['source']['layout_positions_sha256']
                else:assert sigmas==originals['sigmas']
                pipe.load_models_to_device(['dit']);torch.cuda.synchronize();tick=time.perf_counter()
                current=noise[:,:,c['start']:c['stop']].clone();result['status']='sampling';save()
                for j,t in enumerate(pipe.scheduler.timesteps):
                    v=transition_forward(model,current,history=history,history_noise=noise[:,:,:c['start']],
                        full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=float(t)/1000,index=c['index'])
                    current=pipe.scheduler.step(v,t,current);result['sampling_forwards']+=1
                    assert torch.isfinite(current).all()
                    if j%5==0:save();print(f'{args.arm} {c["name"]} {j+1}/30',flush=True)
                torch.cuda.synchronize();sampling=time.perf_counter()-tick
                torch.save(current.cpu(),folder/'endpoint.pt')
                assert th(history)==hh and th(noise)==nh
                pipe.load_models_to_device(['video_vae']);torch.cuda.synchronize();tick=time.perf_counter()
                def decode(z):
                    rgb=pipe.video_vae.decode_video(z,dtype=pipe.torch_dtype,process_image=False,tiled=True,tile_size=256,tile_overlap=64)
                    return pipe.vae_output_to_video(rgb,min_value=0,max_value=1)
                prior=decode(history);assert len(prior)==c['r0']
                visible=decode(torch.cat((history,current),2));assert len(visible)==c['r1']
                frames=visible[c['r0']:];path=folder/'current.mp4';write_video(frames,path)
                zero=decoded(c['zero_video']);new=decoded(path);assert len(zero)==len(new)==len(frames)
                gray=np.stack([np.asarray(f.convert('L'),dtype=np.float32) for f in frames])
                boundary=float(np.abs(gray[0]-np.asarray(prior[-1].convert('L'),dtype=np.float32)).mean())
                sheet=Image.new('RGB',(208*7,138*math.ceil(len(frames)/7)),'white');draw=ImageDraw.Draw(sheet)
                for j,f in enumerate(frames):
                    x=j%7*208;y=j//7*138;draw.text((x+2,y+2),f'RGB{c["r0"]+j}',fill='black');sheet.paste(f.resize((208,120)),(x,y+18))
                sheet.save(folder/'all_frames.jpg',quality=94)
                context=c['context'] if c['context'] is not None else prior[-8:]
                font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);grids=[]
                gt=c['gt'];columns=3 if gt is not None else 2
                for j in range(8+len(frames)):
                    im=Image.new('RGB',(columns*832,528),'black');draw=ImageDraw.Draw(im)
                    panels=([('Actual GT',gt)] if gt is not None else [])+[('Frozen N, 30steps',zero),(f'{args.arm} update4, 30steps, {sampling:.1f}s',new)]
                    for k,(label,video) in enumerate(panels):
                        frame=context[j] if j<8 else video[j-8];im.paste(frame,(k*832,48))
                        draw.text((k*832+8,5),label,font=font,fill='white')
                        draw.text((k*832+8,26),'KNOWN HISTORY' if j<8 else f'{c["name"]} RGB{c["r0"]+j-8}',font=font,fill='white')
                    grids.append(im)
                comparison=folder/'zero_vs_step4.mp4';write_video(grids,comparison)
                flow=flow_metrics(path) if args.suite!='gt' else None
                for check,expected in [(path,len(frames)),(comparison,len(frames)+8)]:
                    with av.open(str(check)) as container:
                        stream=container.streams.video[0];count=sum(1 for _ in container.decode(video=0))
                        assert count==expected and stream.codec_context.name=='h264' and stream.codec_context.pix_fmt=='yuv420p'
                result['records'].append(dict(name=c['name'],history_kind=c['history_kind'],history_hash=hh,noise_hash=nh,
                    prompt_hash=th(prompt),position_hash=th(layout['img_position_ids']),input_sha256=c['expected_input_sha256'],
                    current_RGB_frames=len(frames),RGB_range=[c['r0'],c['r1']],sampling_seconds=sampling,
                    decode_and_artifact_seconds=time.perf_counter()-tick,frame_gray_MAD=float(np.abs(np.diff(gray,axis=0)).mean()),
                    boundary_gray_MAD=boundary,flow=flow,zero_video=str(c['zero_video']),zero_video_sha256=sha(c['zero_video']),
                    current_video=str(path),video_sha256=sha(path),comparison=str(comparison),comparison_sha256=sha(comparison)))
                save();del history,noise,current,anchor,audio
            assert result['sampling_forwards']==60 and result['diagnostic_forwards']==2
            assert all(p._version==v for p,v in versions)
        result.update(status='complete_pending_visual_review',frozen_parameters_unchanged=True,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--arm',choices=['fm_only','fm_action'],required=True)
    ap.add_argument('--suite',choices=['gt','parking_A','parking_D'],required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
