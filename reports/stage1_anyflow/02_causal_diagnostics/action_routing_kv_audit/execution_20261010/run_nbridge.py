"""P2 at middle sigma only: add R1-P/N to separate prefix from video edges.

Requires all six P0 states to have passed. Four new paired forwards plus two
R1/N reproducibility forwards. No video, training, or persistent-cache claim.
"""
import argparse,os,sys,json,time,subprocess
from datetime import datetime
from run import BASE,RT,SOURCE,sha


def main(args):
    gate=json.loads((BASE/'run_04/evaluation.json').read_text())
    assert gate.get('P0_passed') and len(gate['P0'])==6 and all(x['passed'] for x in gate['P0'])
    out=BASE/'N_bridge';out.mkdir(exist_ok=False)
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from audit_core import forward,tensor_hash
    from canonical_sdpa import canonical_attention
    from canonical_gemm import canonical_linears
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    began=time.perf_counter()
    result=dict(status='loading',gpu=args.gpu,pid=os.getpid(),model_forwards=0,records=[],
        optimizer_updates=0,new_videos=0,source_sha256=sha(__file__),
        gate_protocol_sha256=gate['protocol_sha256'],at=datetime.now().astimezone().isoformat())
    def save():
        result['wall_seconds']=time.perf_counter()-began
        p=out/'evaluation.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'evaluation.json')
    save()
    try:
        pipe=abot.load_pipeline('cuda:0')
        released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.eval().requires_grad_(False)
        result['released_LoRA_sha256']=sha(released)
        result['precision']=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        assert result['precision']==gate['precision'] and result['released_LoRA_sha256']==gate['released_LoRA_sha256']
        pipe.load_models_to_device(['dit']);versions=[(p,p._version) for p in model.parameters()]
        data={a:torch.load(SOURCE/f'source_coarse/inputs/parking_{a}.pt',map_location='cpu',weights_only=True) for a in 'AD'}
        def move(x):
            if torch.is_tensor(x):return x.cuda()
            if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
            return x
        packed=move(data['A']['packed']);spans=packed['action_text_spans_local']
        noise=data['A']['initial_noise'].cuda().float();sigma=.689441
        result['status']='P2_N_bridge';save()
        with torch.no_grad():
            for ref in 'AD':
                h=torch.load(SOURCE/f'source_coarse/coarse_A/window0_{ref}.pt',map_location='cpu',weights_only=True).cuda()
                z=torch.load(SOURCE/f'C_{ref}/sigma_noised_{ref}_endpoint.pt',map_location='cpu',weights_only=True).cuda()
                state=(1-sigma)*z+sigma*noise[:,:,12:17];nh=(1-sigma)*h+sigma*noise[:,:,:12]
                values={}
                for action,kind in [('A','R1P'),('D','R1P'),(ref,'R1')]:
                    text=data[ref]['pairs'][0]['prompts'][ref].clone()
                    donor=data[action]['pairs'][0]['prompts'][action]
                    for lo,hi in spans[12:17]:text[lo:hi]=donor[lo:hi]
                    p,t=visible_inputs(packed,text.cuda(),17,390)
                    with canonical_attention(),canonical_linears():
                        v,_=forward(model,state,history=nh,sigma=sigma,kind=kind,history_noised=True,
                            packed=p,prompt=t,anchor=data['A']['anchor'].cuda(),audio=data['A']['audio_noise'].cuda())
                    result['model_forwards']+=1
                    values[kind,action]=v.cpu()
                    result['records'].append(dict(history=ref,action=action,role=kind,sigma=sigma,
                        current_sha256=tensor_hash(state),history_sha256=tensor_hash(nh),
                        output_sha256=tensor_hash(v),prompt_sha256=tensor_hash(t)))
                    save();print(ref,action,kind,flush=True)
                torch.save(values,out/f'{ref}.pt')
            assert all(p._version==v for p,v in versions)
        result['status']='complete'
    except BaseException as exc:result['status']='failed';result['error']=repr(exc);raise
    finally:
        result['GPU_peak_MiB']=torch.cuda.max_memory_allocated()/2**20;save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=4);a=p.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU >=40000MiB')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='True',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
