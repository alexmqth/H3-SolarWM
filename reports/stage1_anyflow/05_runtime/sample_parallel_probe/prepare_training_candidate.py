"""Prepare an isolated trainer; never edit the live or main source trees."""
from pathlib import Path
import difflib
import shutil

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'outputs/2026-10-08-09/stage1_training_shift12/runtime'
DEST=OUT/'training_runtime'
shutil.copytree(SOURCE,DEST,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
shutil.copy2(OUT/'parallel_batch.py',DEST/'code/causal/sample_parallel.py')
p=DEST/'code/causal/train_stage1_anyflow.py'
original=p.read_text();text=original

def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)

replace('import torch\nfrom causal.training_state', 'import torch\nimport torch.distributed as dist\nfrom causal.training_state')
replace('def logical_batch(model, case, chunk, args, *, generator, backward):',
'''def logical_batch(model, case, chunk, args, *, generator, backward):
    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise RuntimeError('Use the four-process candidate entrypoint')
    from causal.sample_parallel import parallel_batch
    return parallel_batch(model, case, chunk, args, generator=generator,
        backward=backward, parameters=[p for p in model.parameters() if p.requires_grad])


def serial_logical_batch(model, case, chunk, args, *, generator, backward):''')
replace("    if (args.out_dir / 'training.json').exists():", "    if dist.get_rank() == 0 and (args.out_dir / 'training.json').exists():")
replace('    def save_result():\n        tmp =', '    def save_result():\n        if dist.get_rank() != 0:\n            return\n        tmp =')
replace('        def checkpoint(directory):\n            directory.mkdir', '        def checkpoint(directory):\n            if dist.get_rank() != 0:\n                return\n            directory.mkdir')
replace("        updates=list(resume['updates']) if resume else [])", """        updates=list(resume['updates']) if resume else [],
        parallel_execution=dict(world_size=4, local_batch=1, global_logical_batch=4,
            gradient_reduction='SUM of already /4 losses', primary_checkpoint_rank=0,
            objective_changed=False, diagonal_shortcut=False,
            note='Runtime-only parallel execution; not a bitwise CUDA serial continuation claim'))""")
replace("('objective', 'scope', 'config', 'history_protocol', 'velocity_convention', 'precision')", "('objective', 'scope', 'config', 'history_protocol', 'velocity_convention', 'precision', 'parallel_execution')")
old='            print(f"[TF-{args.objective}] {step+1}/{args.steps} {case[\'label\']} chunk={chunk} loss={metrics[\'loss\']:.6f} grad={float(norm):.4f}", flush=True)'
replace(old, '            if dist.get_rank() == 0:\n    '+old)
replace('        checkpoint(args.out_dir)\n        result.update(status=', '''        checkpoint(args.out_dir)
        from causal.sample_parallel_state import audit_replicas
        result['replica_audit'] = audit_replicas(params, optimizer, generator,
            device=args.device, smoke=args.smoke)
        result.update(status=''')
p.write_text(text)
(OUT/'training_candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
    fromfile='frozen/train_stage1_anyflow.py',tofile='training_candidate/train_stage1_anyflow.py')))
print(DEST)
