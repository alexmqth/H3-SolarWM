"""Compare outputs/gradients against pinned NVlabs AnyFlow pipeline methods.

Only reviewed inference/scheduler methods are AST-extracted; no external
imports, downloads, model instantiation, optimizer updates or GPU calls.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from flow_map_backward_simulation import simulate_chunk, shortcut_intervals

BASE = Path(__file__).resolve().parent


def methods(file, names, globals_):
    source = file.read_text(); tree = ast.parse(source)
    found = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
             for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {x.name for x in found} == set(names)
    for node in found:
        node.decorator_list = []; node.returns = None
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            arg.annotation = None
    code = ast.fix_missing_locations(ast.Module(body=found, type_ignores=[]))
    ns = dict(globals_); exec(compile(code, str(file), 'exec'), ns)
    return {name: ns[name] for name in names}


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.theta = torch.nn.Parameter(torch.tensor([.2, -.13, .07, .03]))
        self.calls = 0

    def velocity(self, x, t, r):
        self.calls += 1
        return self.theta[0] * torch.tanh(x) + self.theta[1] * t + self.theta[2] * r + self.theta[3] * t * r * x.square()

    def forward(self, hidden_states, timestep, r_timestep, **kwargs):
        t = float(timestep[0, 0]) / 1000
        r = float(r_timestep[0, 0]) / 1000
        return self.velocity(hidden_states, t, r), None


def main():
    torch.set_num_threads(2)
    upstream = BASE / 'upstream'
    pipeline_file = upstream / 'far/pipelines/pipeline_far_wan_anyflow.py'
    scheduler_file = upstream / 'far/schedulers/scheduling_flowmap_euler_discrete.py'
    pinned = {row['path']: row['sha256'] for row in json.loads((BASE / 'source_files.json').read_text())}
    for p in (pipeline_file, scheduler_file):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == pinned[str(p.relative_to(upstream))]
    Scheduler = type('PinnedScheduler', (), methods(scheduler_file,
        ['apply_shift', 'set_timesteps', 'step'], dict(torch=torch, FlowMapEulerDiscreteSchedulerOutput=SimpleNamespace)))
    inference = methods(pipeline_file, ['inference'], dict(torch=torch, copy=copy))['inference']
    Pipeline = type('PinnedPipeline', (), {'inference': inference})
    noise = torch.randn((1, 2, 2, 2, 2), generator=torch.Generator().manual_seed(13))
    records = []
    for shift in (1., 2.22, 5.):
        for budget in (1, 2, 4, 8, 16, 50):
            for index in sorted({0, budget // 2, budget - 1}):
                reference = TinyVelocity(); candidate = TinyVelocity()
                scheduler = Scheduler(); scheduler.config = SimpleNamespace(shift=shift, num_train_timesteps=1000)
                pipe = Pipeline(); pipe.scheduler = scheduler; pipe.transformer = reference
                pipe._execution_device = 'cpu'; pipe.do_classifier_free_guidance = False; pipe.use_mean_velocity = True
                expected = pipe.inference(latents=noise.clone(), num_inference_steps=budget, grad_timestep=index)
                times = [float(x) / 1000 for x in scheduler.timesteps] + [0.]
                actual = simulate_chunk(noise.clone(), candidate.velocity, times, index)
                expected.square().mean().backward(); actual.square().mean().backward()
                error = float((actual - expected).detach().abs().max())
                grad_error = float((candidate.theta.grad - reference.theta.grad).abs().max())
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(candidate.theta.grad, reference.theta.grad, rtol=3e-6, atol=3e-6)
                assert candidate.calls == reference.calls == len(shortcut_intervals(times, index)) <= 3
                records.append(dict(shift=shift, budget=budget, index=index, calls=candidate.calls,
                                    max_output_error=error, max_gradient_error=grad_error))
    # An exact linear ODE map composes analytically; no learned quality claim.
    alpha = torch.tensor(.21, requires_grad=True)
    def exact(x, t, r):
        return x * torch.expm1(alpha * (r - t)) / (r - t)
    simulated = simulate_chunk(noise, exact, [1., .8, .3, 0.], 1)
    analytic = noise * torch.exp(-alpha)
    gs = torch.autograd.grad(simulated.square().mean(), alpha, retain_graph=True)[0]
    ga = torch.autograd.grad(analytic.square().mean(), alpha)[0]
    torch.testing.assert_close(simulated, analytic, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(gs, ga, rtol=2e-6, atol=2e-6)
    # Negative control: detaching the initial shortcut loses real parameter
    # paths. A middle-only replay must not be reported as full FMBS gradient.
    whole = TinyVelocity(); truncated = TinyVelocity(); times = [1., .8, .3, 0.]
    full_output = simulate_chunk(noise, whole.velocity, times, 1)
    with torch.no_grad():
        z = noise + (.8 - 1.) * truncated.velocity(noise, 1., .8)
    for t, r in ((.8, .3), (.3, 0.)):
        z = z + (r - t) * truncated.velocity(z, t, r)
    full_output.square().mean().backward(); z.square().mean().backward()
    gap = float((whole.theta.grad - truncated.theta.grad).norm())
    assert gap > 1e-3
    rejected = 0
    for grid, index in [([1., .5, .5, 0.], 0), ([1., .5, 0.], 2), ([1., .5, .1], 0), ([1., 0.], True)]:
        try:
            shortcut_intervals(grid, index)
        except ValueError:
            rejected += 1
    assert rejected == 4
    result = dict(status='passed', real_H3_training=False, GPU_used=False, upstream_cases=len(records),
                  cases=records, analytic_composition_and_gradient=True,
                  truncated_initial_segment_gradient_gap=gap, invalid_protocols_rejected=rejected,
                  source_commit=json.loads((BASE / 'source.json').read_text())['commit'],
                  max_output_error=max(x['max_output_error'] for x in records),
                  max_gradient_error=max(x['max_gradient_error'] for x in records),
                  scope='CPU algorithm/gradient checks only. No H3 activation-memory, quality, action or on-policy acceptance claim.')
    (BASE / 'fmbs_cpu_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
