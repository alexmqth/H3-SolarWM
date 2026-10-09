"""Isolated E1 C/N history conditioning; frozen T2 attention is unchanged.

C uses immutable clean history and clean video times.
N temporarily interpolates history with fixed exogenous noise and gives all
visible video the current sigma. The caller still integrates CURRENT output
only. No hidden KV is persisted and no previous output/state is overwritten.
"""
import ast
import inspect
import textwrap

from causal.local_topology import window_forward
from single_anchor import original_image_window_variant


def _sigma_history_variant(function):
    """Three audited AST edits: native prefix, fixed I0, no clean-time mask.

    Disabling input_latents_video changes the pipeline's history time override.
    The data input is already the caller's temporally noised history. Solver
    updates remain external and act only on the returned current-window field.
    """
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    original=ast.dump(tree)
    times=[n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='fixed_prefix_timesteps']
    inputs=[n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='input_latents_video']
    retimes=[n for n in ast.walk(tree) if isinstance(n,ast.If)
             and isinstance(n.test,ast.Name) and n.test.id=='start'
             and any(isinstance(c,ast.Attribute) and c.attr=='retime_anchor_position' for c in ast.walk(n))]
    if len(times)!=1 or not isinstance(times[0].value,ast.Constant) or times[0].value.value is not True:
        raise ValueError('Expected one baseline prefix time keyword')
    if len(inputs)!=1 or not isinstance(inputs[0].value,ast.Name) or inputs[0].value.id!='whole':
        raise ValueError('Expected one clean-history input keyword')
    if len(retimes)!=1:
        raise ValueError('Expected one anchor-retiming branch')
    saved=times[0].value,inputs[0].value,retimes[0].test
    times[0].value=ast.Constant(value=False)
    inputs[0].value=ast.Constant(value=None)
    retimes[0].test=ast.Constant(value=False)
    ast.fix_missing_locations(tree)
    namespace={}
    exec(compile(tree,inspect.getfile(function),'exec'),function.__globals__,namespace)
    variant=namespace[function.__name__]
    times[0].value,inputs[0].value,retimes[0].test=saved
    assert ast.dump(tree)==original,'Unregistered AST change'
    return variant


_clean=original_image_window_variant(window_forward)
_noisy=_sigma_history_variant(window_forward)


def history_forward(dit,current,*,history,history_noise,mode,**kwargs):
    """A/D must share history and history_noise; neither may depend on current action."""
    if mode not in ('C','N'):
        raise ValueError('history mode must be C or N')
    if history.shape!=history_noise.shape or history.dtype!=history_noise.dtype or history.device!=history_noise.device:
        raise ValueError('history noise must exactly match the known history layout/dtype/device')
    sigma=float(kwargs['sigma'])
    if not 0 <= sigma <= 1:
        raise ValueError('sigma outside [0,1]')
    rows=(current.shape[-2]//2)*(current.shape[-1]//2)
    if kwargs['anchor'].shape[0]!=rows:
        raise ValueError('Exactly one original RGB image anchor is required')
    if mode=='C':
        return _clean(dit,current,history=history,**kwargs)
    temporary=(1-sigma)*history+sigma*history_noise
    return _noisy(dit,current,history=temporary,**kwargs)
