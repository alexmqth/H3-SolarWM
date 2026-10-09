"""Actual small-H3 loss/gradient/cache/RNG equivalence, including history."""
from datetime import datetime
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNTIME = ROOT / 'outputs/2026-10-08-09/stage1_training_shift12/runtime'
sys.path[:0] = [str(RUNTIME/'code'), str(RUNTIME/'DiffSynth-Studio-h3-v2'),
               str(RUNTIME/'tests')]
import torch
from test_clean_history_graph import setup, detached
from causal.h3_cached import chunk_forward
from causal.clean_history_graph import build_clean_history_graph
from causal.anyflow import anyflow_sample_loss
from diagonal_candidate import sample_loss

torch.set_num_threads(2)
rows = []
for dtype in (torch.float32, torch.bfloat16):
    for full in (False, True):
        model, bank, data, common = setup(dtype)
        clean = data['clean_video'].float()
        noise = data['noise'][:, :, 10:].float()
        cache = detached(model, clean, common)
        saved = {layer: [(x.key.clone(), x.value.clone(), x.rope.clone()) for x in entries]
                 for layer, entries in cache.layers.items()}
        params = bank.parameters()
        for sigma, target in ((.6, .6), (.5731241703033447, .5731241703033447),
                              (1., 1.), (.6, 0.), (.6, .2)):
            results = []
            for function in (anyflow_sample_loss, sample_loss):
                calls = []
                def velocity(x, t, r):
                    grad = torch.is_grad_enabled()
                    calls.append(grad)
                    current_cache = (build_clean_history_graph(model, clean, 2,
                        lambda i: common, checkpoint=True, offload=True)
                        if grad and full else cache)
                    return chunk_forward(model, x, index=2, cache=current_cache,
                        sigma=t, target_sigma=r, allow_grad_read=grad,
                        use_gradient_checkpointing=grad,
                        use_gradient_checkpointing_offload=grad, **common)
                rng = torch.get_rng_state().clone()
                raw, weight, stats = function(velocity, clean[:, :, 10:], noise,
                    sigma, target, shift=12., preserve_fp32_inputs=True)
                gradients = torch.autograd.grad(raw * weight, params, allow_unused=True)
                gradients = [torch.zeros_like(p) if g is None else g
                             for p, g in zip(params, gradients)]
                assert torch.equal(rng, torch.get_rng_state())
                assert all(torch.isfinite(g).all() for g in gradients)
                results.append((raw.detach(), weight, gradients, calls, stats))
            a, b = results
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            assert all(torch.equal(x, y) for x, y in zip(a[2], b[2]))
            assert a[3] == [False, False, False, True]
            assert b[3] == ([True] if sigma == target else a[3])
            for layer, entries in cache.layers.items():
                for entry, values in zip(entries, saved[layer]):
                    assert all(torch.equal(actual, before) for actual, before in
                               zip((entry.key, entry.value, entry.rope), values))
            rows.append(dict(dtype=str(dtype), full_history_gradient=full,
                checkpoint_offload=True, sigma=sigma, target_sigma=target,
                raw_loss=float(a[0]), loss_weight_and_parameter_gradients_exact=True,
                CPU_RNG_unchanged=True, cached_values_unchanged=True,
                reference_current_forwards=len(a[3]), shortcut_current_forwards=len(b[3])))
result = dict(at=datetime.now().astimezone().isoformat(), status='complete',
              scope='CPU mathematical/lifecycle check only; no actual33B throughput or quality claim',
              cases=len(rows), rows=rows)
(OUT/'cpu_equivalence.json').write_text(json.dumps(result, indent=2)+'\n')
print(f'{len(rows)} actual small-H3 cases passed: exact loss, weight, gradients; cache/RNG unchanged')
