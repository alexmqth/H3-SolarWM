"""Single-keyword E1 calibration; frozen topology sources remain untouched.

Original H3 uses native video time for text/action rows. The cached prototype
instead set all text to clean time. Prefix hidden states are recomputed in
T1/T2, so preserving old committed VIDEO KV does not in itself require this
override on CURRENT prefix queries. This module is a diagnostic, not a new
default or a claim that the change improves video quality.
"""
import ast
import inspect
import textwrap


def native_prefix_variant(function):
    """Clone exactly one fixed_prefix_timesteps=True call as False.

    AST editing avoids copying a second evolving attention/window algorithm;
    only this one preprocessing policy changes. No shared global is patched.
    """
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    before=ast.dump(tree)
    changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.keyword) and node.arg=='fixed_prefix_timesteps':
            if not isinstance(node.value,ast.Constant) or node.value.value is not True:
                raise ValueError('Expected the fixed-prefix baseline keyword')
            changed.append(node)
    if len(changed)!=1:
        raise ValueError('Expected exactly one prefix-time policy call')
    changed[0].value=ast.Constant(value=False)
    ast.fix_missing_locations(tree)
    namespace={}
    exec(compile(tree,inspect.getfile(function),'exec'),function.__globals__,namespace)
    variant=namespace[function.__name__]
    changed[0].value=ast.Constant(value=True)
    assert ast.dump(tree)==before,'Unexpected AST change beyond prefix-time policy'
    return variant
