"""Real pretrained VAE prefix audit; no DiT or optimizer."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    torch.set_num_threads(4);torch.manual_seed(13)
    protocol=json.loads((BASE/'protocol.json').read_text())
    result=dict(status='loading',scope=__doc__,gpu=args.gpu,at=datetime.now().astimezone().isoformat(),
        source_sha256=sha(__file__),protocol_sha256=sha(BASE/'protocol.json'),checks=[])
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        p=BASE/'vae_audit.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(BASE/'vae_audit.json')
    save()
    try:
        config=dict(offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,
            onload_device='cpu',preparing_dtype=torch.bfloat16,preparing_device='cuda:0',
            computation_dtype=torch.bfloat16,computation_device='cuda:0')
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda:0',
            model_configs=[ModelConfig(model_id='MiniMax/MiniMax-H3',
                origin_file_pattern='FL2VA/video_vae/source/model.safetensors',**config)],vram_limit=32.)
        assert pipe.dit is None
        pipe.load_models_to_device(['video_vae']);pipe.video_vae.eval().requires_grad_(False)
        def decode(z):
            rgb=pipe.video_vae.decode_video(z.to('cuda:0'),dtype=torch.bfloat16,
                process_image=False,tiled=True,tile_size=256,tile_overlap=64).float().cpu()
            assert torch.isfinite(rgb).all()
            return rgb
        def metrics(a,b):
            d=(a-b).abs();return dict(max_abs=float(d.max()),mean_abs=float(d.mean()))
        with torch.no_grad():
            for ref in 'AD':
                data=torch.load(BASE/f'source_coarse/inputs/parking_{ref}.pt',map_location='cpu',weights_only=True)
                z=data['reference_latents'];full=decode(z)
                for n,nrgb,stable in ((5,17,0),(7,22,17),(10,34,17),(12,39,34),(17,56,51)):
                    prefix=decode(z[:,:,:n]);assert prefix.shape[2]==nrgb
                    row=dict(reference=ref,latent_stop=n,RGB_frames=nrgb,
                        prefix_vs_full=metrics(prefix,full[:,:,:nrgb]),stable_frames=stable,
                        provisional_tail=metrics(prefix[:,:,stable:],full[:,:,stable:nrgb]))
                    if stable:
                        row['stable_prefix']=metrics(prefix[:,:,:stable],full[:,:,:stable])
                        assert row['stable_prefix']['max_abs']==0
                    result['checks'].append(row);save()
                    print(f'[VAE] {ref} prefix{n} RGB{nrgb} stable{stable}',flush=True)
                # Identical full-length decoder graph, perturb future content.
                changed=z.clone();changed[:,:,12:]+=.5*torch.randn_like(changed[:,:,12:])
                other=decode(changed)
                result['checks'].append(dict(reference=ref,intervention='latent12_plus',
                    stable_RGB0_34=metrics(full[:,:,:34],other[:,:,:34]),
                    overlapping_RGB34_39=metrics(full[:,:,34:39],other[:,:,34:39])))
                assert result['checks'][-1]['stable_RGB0_34']['max_abs']==0
                save()
        result.update(status='complete',passed=True,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            note='Real VAE, Original reference latents. No new generation. C freezes prior39 RGB; last5-prefix revision is measured, not silently retroactively applied.')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<34000:raise RuntimeError('VAE audit requires idle GPU')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='True',DIFFSYNTH_MODEL_BASE_PATH=str(RT/'DiffSynth-Studio-h3-v2/models'),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
