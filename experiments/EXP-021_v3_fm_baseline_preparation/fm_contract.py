"""CPU-testable ordinary FM contracts; NOT a 33B V3 training implementation.

The production adapter must supply a differentiable V3 current-block forward.
Existing interval_cached/interval_sw explicitly reject grad-enabled execution.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import torch


def fm_pair(clean: torch.Tensor, noise: torch.Tensor, sigma: float):
    """H3 convention: sigma=0 clean, sigma=1 noise; native t=1000*sigma."""
    if clean.shape != noise.shape or not 0 <= sigma <= 1:
        raise ValueError("matching clean/noise tensors and sigma in [0,1] required")
    return (1 - sigma) * clean + sigma * noise, noise - clean


def visible_actions(actions: list[str], stop: int) -> tuple[str, ...]:
    if not 0 < stop <= len(actions):
        raise ValueError("invalid visible action stop")
    return tuple(actions[:stop])


def detached_clean_commit(commit: Callable, clean_history: torch.Tensor):
    """Only historical clean latents enter a persistent, gradient-free cache."""
    with torch.no_grad():
        result = commit(clean_history.detach())
    if torch.is_tensor(result):
        return result.detach()
    return result


def fm_update(forward: Callable, clean_current: torch.Tensor, noise: torch.Tensor,
              sigma: float, optimizer: torch.optim.Optimizer,
              history_cache, visible_action_prefix: tuple[str, ...]):
    """Interface only: forward must implement V3 attention/conditioning exactly."""
    noisy, target = fm_pair(clean_current, noise, sigma)
    optimizer.zero_grad(set_to_none=True)
    predicted = forward(noisy, sigma, history_cache, visible_action_prefix)
    if predicted.shape != target.shape:
        raise ValueError("velocity shape mismatch")
    loss = torch.nn.functional.mse_loss(predicted.float(), target.float())
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def save_training_state(path: Path, adapter: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, step: int,
                        generator: torch.Generator, metadata: dict):
    if path.exists():
        raise FileExistsError(path)
    torch.save({"adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "logical_rng": generator.get_state(),
                "cpu_rng": torch.get_rng_state(), "metadata": metadata}, path)


def load_training_state(path: Path, adapter: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, generator: torch.Generator,
                        expected_metadata: dict) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["metadata"] != expected_metadata:
        raise ValueError("checkpoint protocol metadata mismatch")
    adapter.load_state_dict(state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["logical_rng"])
    torch.set_rng_state(state["cpu_rng"])
    return int(state["step"])
