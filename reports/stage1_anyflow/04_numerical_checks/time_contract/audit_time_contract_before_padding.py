"""CPU audit of actual H3 time weights and production packed time routing.

Intercept at the transformer boundary; no transformer, attention, KV, VAE,
optimizer, or GPU execution. This prepares C; it does not establish a credible
causal checkpoint or satisfy the trained-model diagonal/quality requirements.
"""
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from safetensors import safe_open

OUT = Path(__file__).resolve().parent
H3 = OUT.parents[2]
ROOT = H3.parent
sys.path[:0] = [str(H3/'code'), str(H3/'DiffSynth-Studio-h3-v2')]
from causal.anyflow import H3AnyFlowConditioner
from causal.anyflow_sampling import configure_video_schedule
from causal.h3_cached import H3ChunkCache, chunk_forward, slice_packed
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.models.minimax_h3_dit import MiniMaxH3TimeEmbedder
from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Captured(Exception):
    pass


class BoundaryCapture:
    _h3_input_dtype = torch.float32
    anyflow_conditioner = True

    def __call__(self, **kwargs):
        self.last = kwargs
        raise Captured()


def invoke(function):
    try:
        function()
    except Captured:
        return
    raise AssertionError('Expected capture before any transformer execution')


def main():
    destination = OUT/'audit.json'
    if destination.exists():
        raise FileExistsError('Inspect existing audit rather than overwrite')
    torch.set_num_threads(4)
    torch.manual_seed(13)
    started = time.perf_counter()
    report = dict(status='running', at=datetime.now().astimezone().isoformat(),
        scope='actual FP32 time weights; production preprocessing intercepted before transformer',
        GPU_forwards=0, transformer_forwards=0, optimizer_updates=0,
        C_complete=False, trained_diagonal_preservation='not measured', records=[])
    try:
        native = H3/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer'
        index = json.loads((native/'model.safetensors.index.json').read_text())['weight_map']
        weights = {}
        for name in ('proj_in.weight', 'proj_in.bias', 'proj_out.weight', 'proj_out.bias'):
            key = 'time_embedder.'+name
            with safe_open(native/index[key], framework='pt', device='cpu') as file:
                weights[name] = file.get_tensor(key)
            assert weights[name].dtype == torch.float32
        base = MiniMaxH3TimeEmbedder(weights['proj_in.weight'].shape[1],
            weights['proj_in.weight'].shape[0], weights['proj_out.weight'].shape[0]).eval()
        base.load_state_dict(weights, strict=True)
        conditioner = H3AnyFlowConditioner(base, gate=.25).eval()
        assert all(torch.equal(base.state_dict()[k], conditioner.delta.state_dict()[k]) for k in weights)
        report['time_weights'] = {k:dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_sha(v)) for k,v in weights.items()}
        lora_path = H3/'checkpoints/H3-World/step-10000.safetensors'
        with safe_open(lora_path, framework='pt', device='cpu') as file:
            time_keys = [key for key in file.keys() if 'time_embedder' in key]
        assert not time_keys
        report['released_action_lora_time_keys'] = time_keys
        # Execute SolarWM's real mixing implementation with the same loaded
        # integrated time embedding. Identity separates no additional scale.
        official = ROOT/'SolarWM/src/solarwm/backends/minimax_h3/anyflow_conditioning.py'
        spec = importlib.util.spec_from_file_location('official_h3_time', official)
        source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
        class Reference(source.H3AnyFlowConditioningMixin, nn.Module):
            def __init__(self):
                super().__init__(); self.time_proj = nn.Identity(); self.time_embedder = Embedding(base)
        class Embedding(nn.Module):
            def __init__(self, module):
                super().__init__(); self.module=module
            def forward(self, t):
                return self.module(t, dtype=torch.float32)
        reference = Reference().eval(); reference.enable_anyflow(.25)
        report['sources'] = {str(p.relative_to(ROOT)):sha(p) for p in (
            Path(__file__), official, H3/'code/causal/anyflow.py',
            H3/'code/causal/anyflow_sampling.py', H3/'code/causal/h3_cached.py',
            H3/'DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py',
            H3/'DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py')}
        manifest = json.loads((H3/'data/abot_bridge/encoded_manifest.json').read_text())
        record = next(r for r in manifest['clips'] if r['clip_id']=='118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140')
        assert sha(record['encoded_file']) == record['sha256']
        raw = torch.load(record['encoded_file'], map_location='cpu', weights_only=True)
        cond = raw['conditioning']; clean = raw['clean_latents'].float()
        report['data_source'] = dict(clip=record['clip_id'], encoded_sha256=record['sha256'])
        capture = BoundaryCapture()
        max_diagonal_error = 0.; max_reference_error = 0.
        unique_embeddings = {}

        def check(captured, *, tag, sigma, rho, anchor_rows, history_frames=0):
            nonlocal max_diagonal_error, max_reference_error
            times = captured['unique_timesteps']; targets = captured['unique_target_timesteps']
            inv = captured['inverse_indices']; current_rows = times[inv]; target_rows = targets[inv]
            assert torch.isfinite(times).all() and torch.isfinite(targets).all()
            assert bool(((times>=0)&(times<=1)&(targets>=0)&(targets<=1)).all())
            video = captured['img_pos_info']['position_ids']; anchors = video[:anchor_rows]
            video = video[anchor_rows:]; rows_per_frame=captured['action_frame_rows']
            history = video[:history_frames*rows_per_frame]; active=video[history_frames*rows_per_frame:]
            text = captured['text_pos_info']['position_ids']; audio = captured['audio_pos_info']['position_ids']
            groups = dict(text=text, RGB_anchors=anchors, audio=audio, clean_history=history, current_video=active)
            native_t=1-float(torch.tensor(sigma*1000))/1000
            native_r=1-float(torch.tensor(rho*1000))/1000
            expected=dict(text=(1.,1.),RGB_anchors=(.999,.999),audio=(0.,0.),
                clean_history=(1.,1.),current_video=(native_t,native_r))
            for label,positions in groups.items():
                a,b=expected[label]
                assert torch.equal(current_rows[positions], torch.full_like(current_rows[positions],a)), (tag,label,'current')
                assert torch.equal(target_rows[positions], torch.full_like(target_rows[positions],b)), (tag,label,'target')
            embedded=base(times,dtype=torch.float32)
            mixed=conditioner(embedded,targets,dtype=torch.float32)
            expected_mix=reference._time_condition(times,targets,parameter_dtype=lambda _:torch.float32)
            reference_error=float((mixed-expected_mix).abs().max())
            assert reference_error==0.
            max_reference_error=max(max_reference_error,reference_error)
            if sigma==rho:
                error=float((mixed-embedded).abs().max());max_diagonal_error=max(max_diagonal_error,error)
                torch.testing.assert_close(mixed,embedded,rtol=1e-6,atol=1e-5)
            for a,b in zip(times.tolist(),targets.tolist()):
                if (a,b) not in unique_embeddings:
                    ta=torch.tensor([a]);tb=torch.tensor([b]);ea=base(ta,dtype=torch.float32);eb=base(tb,dtype=torch.float32)
                    em=conditioner(ea,tb,dtype=torch.float32)
                    unique_embeddings[a,b]=dict(current_native=a,target_native=b,current_norm=float(ea.norm()),
                        target_norm=float(eb.norm()),mixed_norm=float(em.norm()),
                        mixed_relative_change=float((em-ea).norm()/ea.norm()),mixed_sha256=tensor_sha(em))
            report['records'].append(dict(tag=tag,sigma=sigma,rho=rho,all_rows_verified=True,
                token_counts={k:int(v.numel()) for k,v in groups.items()},
                unique_pairs=list(map(list,zip(times.tolist(),targets.tolist()))),
                current_row_sha256=tensor_sha(current_rows),target_row_sha256=tensor_sha(target_rows),
                official_mix_max_abs_error=reference_error))
            if sigma==1 and rho<1:
                assert len(times)>len(torch.unique(times)), 'Audio/video same t must retain different target times'

        with torch.no_grad():
            for grid in ('native','uniform'):
                for steps in (4,8):
                    schedule=configure_video_schedule(FlowMatchScheduler('MiniMax-H3'),steps=steps,grid=grid,flow_shift=2.22)
                    for chunk in (0,1,2):
                        start=chunk*5;current=cond['initial_noise'][:,:,start:min(start+5,12)].float()
                        anchor=raw['anchors'][chunk].float()
                        for interval,(t,r) in enumerate(zip(schedule,schedule[1:])):
                            for rho in (t,r):
                                invoke(lambda:chunk_forward(capture,current,full_packed=raw['causal_packed'],
                                    prompt=cond['prompt_embeds'],anchor=anchor,audio=cond['audio_noise'],
                                    sigma=t,target_sigma=rho,index=chunk,chunk_frames=5,cache=H3ChunkCache(5,'cpu'),
                                    anchor_frame_index=start-1 if chunk else None,anchor_slot=1,
                                    action_prefix_mode='causal',action_feedback=True))
                                check(capture.last,tag=f'{grid}/{steps}/chunk{chunk}/interval{interval}/'+('diagonal' if rho==t else 'finite'),
                                    sigma=t,rho=rho,anchor_rows=anchor.shape[0])
            # Full clean-history packing and clean commit endpoints use the
            # same production preprocessing, without evaluating their KV.
            rows=(clean.shape[-2]//2)*(clean.shape[-1]//2)
            for t,r in ((1.,0.),(.24078089904785155,0.),(.6894410400390625,.24078089904785155),(0.,0.)):
                mask=torch.ones_like(clean);mask[:,:,:10]=0
                anchor=raw['anchors'][2].float()
                invoke(lambda:model_fn_minimax_h3(capture,clean,cond['audio_noise'].float(),
                    slice_packed(raw['causal_packed'],0,12,rows),cond['prompt_embeds'],
                    timestep_video=torch.tensor(t*1000),timestep_audio=torch.tensor(1000.),
                    target_timestep_video=torch.tensor(r*1000),keyframe_cond_anchor=anchor,
                    input_latents_video=clean,denoise_mask_video=mask,fixed_prefix_timesteps=True))
                check(capture.last,tag='full_prefix/history10',sigma=t,rho=r,anchor_rows=anchor.shape[0],history_frames=10)
            x=torch.tensor([.25,.5,.75,1.])
            native_embedding=base(x,dtype=torch.float32)
            wrong_scaled=base(x*1000,dtype=torch.float32)
            scale_error=float((wrong_scaled-native_embedding).norm()/native_embedding.norm())
            assert scale_error>.01
        assert len(report['records'])==148
        report.update(status='passed',row_contract_cases=len(report['records']),
            embeddings=sorted(unique_embeddings.values(),key=lambda r:(r['current_native'],r['target_native'])),
            max_official_mix_abs_error=max_reference_error,max_initial_diagonal_embedding_abs_error=max_diagonal_error,
            wrong_1000_scale_relative_error=scale_error,
            limitation='Production preprocessing and actual time MLP only. No transformer velocity/attention/KV evaluated; no trained diagonal or visual/action PASS.')
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['wall_seconds']=time.perf_counter()-started
        destination.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('records','embeddings','time_weights','sources')},indent=2),flush=True)


if __name__=='__main__':
    main()
