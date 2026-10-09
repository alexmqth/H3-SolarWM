"""Compare the actual installed native Diffusers time projection with our port.

Execute unmodified AST nodes from the SolarWM environment, avoiding import of
the 33B transformer and its CUDA dependencies. No training/model changes.
"""
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from safetensors import safe_open
from diffusers.models.activations import get_activation

OUT=Path(__file__).resolve().parent
H3=OUT.parents[2]
sys.path[:0]=[str(H3/'DiffSynth-Studio-h3-v2')]
from diffsynth.models.minimax_h3_dit import MiniMaxH3TimeEmbedder


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target=OUT/'native_time_comparison.json'
    if target.exists():raise FileExistsError('Preserve existing comparison')
    torch.set_num_threads(4)
    directory=Path('/home/lpeng/miniconda3/envs/solarwm-h3/lib/python3.10/site-packages/diffusers/models')
    embeddings=directory/'embeddings.py'
    transformer=directory/'transformers/transformer_minimax_h3.py'
    names={'get_timestep_embedding','Timesteps','TimestepEmbedding'}
    nodes=[n for n in ast.parse(embeddings.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==names
    namespace=dict(torch=torch,nn=nn,math=math,get_activation=get_activation)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(embeddings),'exec'),namespace)
    # Inspect the actual native constructor policy, not a guessed default.
    assignments=[n for n in ast.walk(ast.parse(transformer.read_text())) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr=='time_proj']
    assert len(assignments)==1
    call=assignments[0].value
    assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='Timesteps'
    assert {k.arg:ast.unparse(k.value) for k in call.keywords}==dict(num_channels='freq_dim',flip_sin_to_cos='True',downscale_freq_shift='0')
    native=H3/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer'
    index=json.loads((native/'model.safetensors.index.json').read_text())['weight_map']
    weights={}
    for suffix in ('proj_in.weight','proj_in.bias','proj_out.weight','proj_out.bias'):
        name='time_embedder.'+suffix
        with safe_open(native/index[name],framework='pt',device='cpu') as f:weights[suffix]=f.get_tensor(name)
    fi=weights['proj_in.weight'].shape[1];hi=weights['proj_in.weight'].shape[0];ou=weights['proj_out.weight'].shape[0]
    port=MiniMaxH3TimeEmbedder(fi,hi,ou).eval();port.load_state_dict(weights,strict=True)
    projection=namespace['Timesteps'](fi,flip_sin_to_cos=True,downscale_freq_shift=0)
    mlp=namespace['TimestepEmbedding'](fi,hi,out_dim=ou).eval()
    assert isinstance(mlp.act,nn.SiLU) and mlp.cond_proj is None and mlp.post_act is None
    mlp.load_state_dict({k.replace('proj_in','linear_1').replace('proj_out','linear_2'):v for k,v in weights.items()},strict=True)
    previous=json.loads((OUT/'audit.json').read_text())
    for k,w in weights.items():assert hashlib.sha256(w.numpy().tobytes()).hexdigest()==previous['time_weights'][k]['sha256']
    values=sorted({r[k] for r in previous['embeddings'] for k in ('current_native','target_native')})
    with torch.no_grad():
        times=torch.tensor(values,dtype=torch.float32)
        expected=mlp(projection(times));actual=port(times,dtype=torch.float32)
        error=float((expected-actual).abs().max())
        assert torch.equal(expected,actual)
    result=dict(status='passed',at=datetime.now().astimezone().isoformat(),
        sources={str(p):sha(p) for p in (Path(__file__),embeddings,transformer)},
        extracted_AST_nodes=sorted(names),AST_bodies_modified=False,
        native_constructor=ast.unparse(assignments[0]),time_values=values,
        compared_native_times=len(values),maximum_abs_error=error,bit_identical=True,
        actual_weight_shapes={k:list(v.shape) for k,v in weights.items()},
        scope='CPU actual native sinusoid and MLP versus DiffSynth port, FP32 initialization only',
        transformer_forwards=0,optimizer_updates=0,C_complete=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
