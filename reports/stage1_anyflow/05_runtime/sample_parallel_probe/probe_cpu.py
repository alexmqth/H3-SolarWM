"""Real four-process Gloo probe against serial small-H3 logical_batch."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNTIME = ROOT / 'outputs/2026-10-08-09/stage1_training_shift12/runtime'
sys.path[:0] = [str(RUNTIME/'code'), str(RUNTIME/'DiffSynth-Studio-h3-v2'), str(RUNTIME/'tests')]
import torch
import torch.distributed as dist
import causal.train_stage1_anyflow as trainer
from test_clean_history_graph import setup
from parallel_batch import parallel_batch

torch.set_num_threads(2)
dist.init_process_group('gloo', timeout=timedelta(minutes=5))
rank = dist.get_rank()
results = []
try:
    cases = [('anyflow','full',i,torch.float32) for i in (0,1,2)]
    cases += [('anyflow','detached',2,torch.float32), ('fm','full',2,torch.float32),
              ('anyflow','full',2,torch.bfloat16)]
    for objective, history, chunk, dtype in cases:
        model, bank, data, common = setup(dtype, anyflow=objective=='anyflow')
        parameters = bank.parameters()
        case = dict(label='synthetic', clean=data['clean_video'].float(), packed=data['packed'],
            prompt=common['prompt'], audio=common['audio'], anchors=[common['anchor']]*3,
            actions=None, action_adapter=None)
        args = SimpleNamespace(logical_batch=4, chunk_frames=5, history_chunks=5,
            objective=objective, precision_profile='h3_fp32', flow_shift=2.22,
            training_timestep_shift=12., validation_timestep_shift=2.22,
            finite_difference_epsilon=5., history_gradient_mode=history, anchor_mode='fixed',
            action_prefix_mode='causal', action_feedback=True, smoke=True)
        rng = torch.get_rng_state().clone()
        reference_noise = []
        original_loss = trainer.anyflow_sample_loss
        def capture(velocity, clean, noise, *args, **kwargs):
            reference_noise.append(hashlib.sha256(noise.detach().float().cpu().numpy().tobytes()).hexdigest())
            return original_loss(velocity, clean, noise, *args, **kwargs)
        trainer.anyflow_sample_loss = capture
        serial_generator = torch.Generator().manual_seed(1001)
        reference = trainer.logical_batch(model, case, chunk, args,
            generator=serial_generator, backward=True)
        trainer.anyflow_sample_loss = original_loss
        reference_gradient = [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in parameters]
        assert torch.equal(rng, torch.get_rng_state())
        for p in parameters:
            p.grad = None
        parallel_generator = torch.Generator().manual_seed(1001)
        actual = parallel_batch(model, case, chunk, args, generator=parallel_generator,
            backward=True, parameters=parameters, record_noise=True)
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(serial_generator.get_state(), parallel_generator.get_state())
        assert actual['samples'] == reference['samples']
        assert actual['loss'] == reference['loss']
        assert actual['model_evaluations'] == reference['model_evaluations']
        assert actual['differentiable_history_forwards'] == reference['differentiable_history_forwards']
        if objective == 'anyflow':
            assert reference_noise == [r['actual_noise_sha256'] for r in actual['ranks']]
        for p, expected in zip(parameters, reference_gradient):
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, expected, atol=2e-7, rtol=2e-5)
        difference = float(torch.sqrt(sum((p.grad.double()-g.double()).square().sum()
                                         for p,g in zip(parameters,reference_gradient))))
        # One real Adam update from equal starting parameters and state,
        # compared to the serial gradient; checks against accidental /world.
        expected_params = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        serial_optimizer = torch.optim.AdamW(expected_params, lr=3e-5, betas=(.9,.95), weight_decay=.01)
        parallel_optimizer = torch.optim.AdamW(parameters, lr=3e-5, betas=(.9,.95), weight_decay=.01)
        for p, g in zip(expected_params, reference_gradient):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(expected_params,1.)
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        serial_optimizer.step(); parallel_optimizer.step()
        for p, expected in zip(parameters, expected_params):
            torch.testing.assert_close(p, expected, atol=2e-7, rtol=2e-5)
        results.append(dict(objective=objective, history=history, chunk=chunk, dtype=str(dtype),
            raw_samples_exact=True, actual_AnyFlow_noise_exact=objective=='anyflow',
            CPU_and_logical_RNG_exact=True, gradient_difference_norm=difference,
            one_Adam_update_close=True, global_batch=4, ranks=4))
        if rank==0:
            print(results[-1], flush=True)
    gathered = [None]*4
    dist.all_gather_object(gathered, results)
    assert all(x==results for x in gathered)
    if rank==0:
        (OUT/'cpu_equivalence.json').write_text(json.dumps(dict(status='complete',
            at=datetime.now().astimezone().isoformat(), backend='gloo', world_size=4,
            scope='actual four-process small-H3 CPU gradient and Adam check; no33B/GPU/quality evidence',
            cases=results),indent=2)+'\n')
finally:
    dist.destroy_process_group()
