"""CPU execution audit of the completed T2 history/time contract, no new method."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
RT=BASE/'runtime'
sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2')]
import torch
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import window_forward,visible_inputs
from single_anchor import original_image_window_variant
from diffsynth.pipelines.minimax_h3_audio_video import patchify_video


def main():
    torch.set_num_threads(2)
    torch.manual_seed(911)
    model=make_small_h3().eval().requires_grad_(False)
    configure_precision(model,'h3_fp32')
    b=synthetic_h3_batch(frames=37,seed=321)
    single=original_image_window_variant(window_forward)
    captured=[]
    def capture(module,args,kwargs):
        captured.append(kwargs)
    hook=model.register_forward_pre_hook(capture,with_kwargs=True)
    records=[]
    try:
        with torch.no_grad():
            for i,start in enumerate((0,12,24)):
                stop=start+12
                packed,text=visible_inputs(b['packed'],b['prompt_embeds'],stop,4)
                history=b['clean_video'][:,:,:start]
                state=b['noise'][:,:,start:stop]
                for sigma in (1.,.5,.1):
                    prior=history.clone()
                    single(model,state,history=history,full_packed=packed,prompt=text,
                           anchor=b['anchor_rows'],audio=b['audio_latents'],sigma=sigma,
                           index=i,chunk_frames=12,history_chunks=5)
                    kw=captured.pop()
                    times=kw['unique_timesteps'][kw['inverse_indices']]
                    pos=kw['img_pos_info']['position_ids'].flatten()
                    video=pos[4:]
                    assert len(video)==stop*4
                    hist_rows,cur_rows=video[:start*4],video[start*4:]
                    assert torch.equal(history,prior)
                    torch.testing.assert_close(kw['x'][0,video],patchify_video(torch.cat([history,state],2)),atol=0,rtol=0)
                    assert torch.equal(times[hist_rows],torch.ones_like(times[hist_rows]))
                    torch.testing.assert_close(times[cur_rows],torch.full_like(times[cur_rows],1-sigma),atol=1e-7,rtol=0)
                    text_pos=kw['text_pos_info']['position_ids'].flatten()
                    torch.testing.assert_close(times[text_pos],torch.full_like(times[text_pos],1-sigma),atol=1e-7,rtol=0)
                    action=[]
                    for frame,(lo,hi) in enumerate(packed['action_text_rows'].tolist()):
                        target=times[lo:hi]
                        torch.testing.assert_close(target,torch.full_like(target,1-sigma),atol=1e-7,rtol=0)
                        action.append(dict(frame=frame,time=float(target[0]),video_time=1. if frame<start else 1-sigma))
                    records.append(dict(window=i,latent_range=[start,stop],sigma=sigma,
                                        history_input='clean immutable latent',history_video_time=1. if start else None,
                                        current_video_time=1-sigma,all_action_time=1-sigma,
                                        history_action_video_time_gap=sigma if start else None,
                                        future_action_rows=0,per_frame=action))
    finally:
        hook.remove()
    result=dict(at=datetime.now().astimezone().isoformat(),status='complete_cpu_contract_audit',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records,
                model='actual tiny-H3 executing frozen pipeline, synthetic37 layout; not a33B effect experiment',
                optimizer_updates=0,GPU_calls=0,
                interpretation='The current T2 mixes clean historical video with current-sigma action/text times. This follows the existing pipeline denoise-mask semantics; it is not evidence of an alignment bug or proof of failure cause. Original retake also supports clean video rows. Hypotheses require controlled effects.',
                next_constraint='Do not silently change past-action times or re-noise history in completed runs. Register any next conditioning intervention separately.')
    (BASE/'history_contract_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],states=len(records),GPU_calls=0)))


if __name__=='__main__':
    main()
