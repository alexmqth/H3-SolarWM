"""Same AnyFlow128 weights, diagonal (r=t) velocity sampling; diagnostic only."""
import importlib.util
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNTIME = ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128/training_runtime'
sys.path.insert(0,str(RUNTIME/'code'))
spec=importlib.util.spec_from_file_location('diagonal_benchmark',RUNTIME/'code/causal/benchmark.py')
benchmark=importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
original_forward, original_save = benchmark.chunk_forward, benchmark.save_json
trace=[]

def diagonal_forward(dit,current,**kwargs):
    if getattr(dit,'anyflow_conditioner',None) is None:
        raise RuntimeError('This diagnostic requires the actual AnyFlow model')
    sigma=float(kwargs['sigma'])
    requested=kwargs.get('target_sigma')
    commit=bool(kwargs.get('commit',False))
    if commit and sigma != 0.0:
        raise RuntimeError('Unexpected non-clean cache commit')
    trace.append(dict(chunk=kwargs['index'],commit=commit,sigma=sigma,
        integration_target_sigma=None if requested is None else float(requested),
        model_target_sigma=sigma))
    kwargs['target_sigma']=sigma
    return original_forward(dit,current,**kwargs)

def save_json(path,data):
    copy=dict(data)
    copy['inference_diagnostic']='same_weights_diagonal_r_equals_t_velocity_euler'
    if path.name=='cached.json':
        copy['sampler']='anyflow_diagonal_velocity_euler'
        copy['model_time_pairs']=list(trace)
        copy['integration_grid_unchanged']=True
        if copy.get('status')=='complete':
            if len(trace)!=27 or sum(row['commit'] for row in trace)!=3:
                raise RuntimeError('Unexpected denoiser/commit counts')
            if not all(row['model_target_sigma']==row['sigma'] for row in trace):
                raise RuntimeError('Diagonal condition was not preserved')
    original_save(path,copy)

benchmark.chunk_forward=diagonal_forward
benchmark.save_json=save_json
if __name__=='__main__':
    benchmark.main()
