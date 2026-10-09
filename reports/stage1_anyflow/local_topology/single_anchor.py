"""Isolated Original-I0 calibration: native text time, no anchor retiming.

Never move the original image to a previous chunk's timestamp. Frozen E1
runtime stays unchanged. Only the declared two AST changes are permitted.
"""
import ast
import functools
import inspect
import textwrap


def original_image_window_variant(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    before = ast.dump(tree)
    times = [n for n in ast.walk(tree) if isinstance(n, ast.keyword)
             and n.arg == 'fixed_prefix_timesteps']
    retimes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
               and isinstance(n.test, ast.Name) and n.test.id == 'start'
               and any(isinstance(c, ast.Attribute) and c.attr == 'retime_anchor_position'
                       for c in ast.walk(n))]
    if len(times) != 1 or not isinstance(times[0].value, ast.Constant) or times[0].value.value is not True:
        raise ValueError('Expected one fixed-prefix baseline call')
    if len(retimes) != 1:
        raise ValueError('Expected one dynamic-anchor retime branch')
    old_time, old_test = times[0].value, retimes[0].test
    times[0].value = ast.Constant(value=False)
    retimes[0].test = ast.Constant(value=False)
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getfile(function), 'exec'), function.__globals__, namespace)
    variant = namespace[function.__name__]
    times[0].value, retimes[0].test = old_time, old_test
    assert ast.dump(tree) == before, 'Unexpected change beyond time and I0 position policy'

    @functools.wraps(function)
    def single(dit, current, **kwargs):
        rows = (current.shape[-2] // 2) * (current.shape[-1] // 2)
        if kwargs['anchor'].shape[0] != rows:
            raise ValueError('Exactly one original RGB image anchor is required')
        return variant(dit, current, **kwargs)
    return single
