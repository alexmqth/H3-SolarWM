"""Keep native sampling unchanged and verify the official four-map grid."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from causal.anyflow_sampling import configure_video_schedule
from causal.anyflow import finite_map_step

spec = importlib.util.spec_from_file_location('h3_schedule_oracle',
    ROOT / 'DiffSynth-Studio-h3-v2/diffsynth/diffusion/flow_match.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Scheduler = module.FlowMatchScheduler


@pytest.mark.parametrize('steps', [4, 8])
def test_native_is_exactly_legacy(steps):
    expected, actual = Scheduler('MiniMax-H3'), Scheduler('MiniMax-H3')
    expected.set_timesteps(steps, shift=2.22)
    configure_video_schedule(actual, steps=steps, flow_shift=2.22)
    assert torch.equal(expected.timesteps, actual.timesteps)
    assert torch.equal(expected.sigmas, actual.sigmas)


def test_official_grid_and_complete_finite_map_trajectory():
    scheduler = Scheduler('MiniMax-H3')
    rng = torch.random.get_rng_state().clone()
    grid = configure_video_schedule(scheduler, steps=4, grid='uniform', flow_shift=12.)
    assert grid == [1., .75, .5, .25, 0.]
    assert torch.equal(rng, torch.random.get_rng_state())
    # A constant oracle noise-clean velocity must reach clean after all four
    # intervals. This catches omitted/doubled final endpoints or sign errors.
    clean, noise = torch.tensor([2., -3.]), torch.tensor([7., 9.])
    current = noise.clone()
    for sigma, target in zip(grid[:-1], grid[1:]):
        current = finite_map_step(current, noise - clean, sigma, target)
    assert torch.equal(current, clean)
    assert not scheduler.training


def test_uniform_grid_does_not_change_with_training_shift():
    one, two = Scheduler('MiniMax-H3'), Scheduler('MiniMax-H3')
    assert configure_video_schedule(one, steps=8, grid='uniform', flow_shift=2.22) == configure_video_schedule(two, steps=8, grid='uniform', flow_shift=12.)
