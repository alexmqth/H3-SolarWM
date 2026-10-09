"""Stage1 LoRA on all H3 block/refiner Q/K/V/out/FFN projections.

Keep the existing visual/action initialization frozen underneath this bank.
Separate Q/K/V low-rank factors reproduce the official split-projection
capacity even though DiffSynth executes one fused QKV projection.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SUFFIXES = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')


def linear_shape(module):
    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
        return int(module.in_features), int(module.out_features)
    if hasattr(module, 'base'):
        return linear_shape(module.base)
    raise TypeError(f'Expected materialized linear or frozen LoRA wrapper: {type(module)}')


class Stage1LinearLoRA(nn.Module):
    def __init__(self, base, *, rank, alpha, device, split_qkv=False):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA rank/alpha must be positive and finite')
        self.in_features, self.out_features = linear_shape(base)
        self.parts = 3 if split_qkv else 1
        if self.out_features % self.parts:
            raise ValueError('Fused QKV output must split into equal Q/K/V projections')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = self.alpha / self.rank
        self.enabled = True
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=torch.float32))
            for _ in range(self.parts)])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features // self.parts, rank,
                                     device=device, dtype=torch.float32))
            for _ in range(self.parts)])
        for weight in self.lora_A:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.base(x)
        if not self.enabled:
            return base_output
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = torch.cat([F.linear(F.linear(x.float(), a), b)
                               for a, b in zip(self.lora_A, self.lora_B)], dim=-1)
        return base_output + (delta * self.scale).to(base_output.dtype)

    def adapter_parameters(self):
        return [p for pair in zip(self.lora_A, self.lora_B) for p in pair]


def target_names(model):
    prefixes = [f'blocks.{i}' for i in range(len(model.blocks))]
    prefixes += [f'token_refiner.blocks.{i}' for i in range(len(model.token_refiner.blocks))]
    if not prefixes:
        raise ValueError('No H3 block/refiner targets found')
    names = tuple(f'{prefix}.{suffix}' for prefix in prefixes for suffix in SUFFIXES)
    for name in names:
        linear_shape(model.get_submodule(name))
    return names


@dataclass
class Stage1LoRABank:
    names: tuple[str, ...]
    modules: list[Stage1LinearLoRA]
    rank: int
    alpha: float

    def parameters(self, group=None):
        def matches(name):
            if group is None:
                return True
            if group == 'qkv':
                return name.endswith('attn.qkv_proj')
            if group == 'out':
                return name.endswith('attn.out_proj')
            if group == 'ffn':
                return '.mlp.' in name
            if group == 'refiner':
                return name.startswith('token_refiner.')
            raise ValueError(f'Unknown adapter group {group}')
        return [p for name, module in zip(self.names, self.modules)
                if matches(name) for p in module.adapter_parameters()]

    def describe(self):
        return dict(scope='all_qkvo_ffn', rank=self.rank, alpha=self.alpha,
                    fused_wrappers=len(self.names),
                    logical_linear_targets=sum(m.parts for m in self.modules),
                    trainable_parameters=sum(p.numel() for p in self.parameters()),
                    targets=list(self.names),
                    initialization='existing visual/action frozen; added B=0; independent split Q/K/V')


def install_stage1_lora(model, *, rank=8, alpha=8., device='cpu'):
    names = target_names(model)
    if any(isinstance(model.get_submodule(n), Stage1LinearLoRA) for n in names):
        raise ValueError('Stage1 LoRA bank is already installed')
    modules = []
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == 'cuda' else []
    # Adding trainable capacity must not change subsequent training/video noise.
    with torch.random.fork_rng(devices=devices):
        for name in names:
            parent, _, slot = name.rpartition('.')
            module = Stage1LinearLoRA(model.get_submodule(name), rank=rank,
                alpha=alpha, device=device, split_qkv=name.endswith('attn.qkv_proj'))
            setattr(model.get_submodule(parent), slot, module)
            modules.append(module)
    return Stage1LoRABank(names, modules, int(rank), float(alpha))


def save_stage1_lora(path, bank, metadata):
    path = Path(path)
    state = dict(format='h3world_stage1_all_qkvo_ffn_v1', rank=bank.rank, alpha=bank.alpha,
        targets=list(bank.names),
        weights={name: dict(A=[p.detach().cpu() for p in module.lora_A],
                            B=[p.detach().cpu() for p in module.lora_B])
                 for name, module in zip(bank.names, bank.modules)}, metadata=metadata)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    temporary.replace(path)


def load_stage1_lora(model, path, *, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'h3world_stage1_all_qkvo_ffn_v1':
        raise ValueError('Unsupported Stage1 full-scope adapter format')
    names = target_names(model)
    if list(names) != state['targets'] or set(names) != set(state['weights']):
        raise ValueError('Stage1 adapter targets do not match the H3 topology')
    # Validate all shapes before mutating the model.
    for name in names:
        inputs, outputs = linear_shape(model.get_submodule(name))
        parts = 3 if name.endswith('attn.qkv_proj') else 1
        values = state['weights'][name]
        if (len(values['A']) != parts or len(values['B']) != parts
                or any(p.shape != (state['rank'], inputs) for p in values['A'])
                or any(p.shape != (outputs // parts, state['rank']) for p in values['B'])):
            raise ValueError(f'Stage1 adapter shape mismatch: {name}')
    bank = install_stage1_lora(model, rank=state['rank'], alpha=state['alpha'], device=device)
    with torch.no_grad():
        for name, module in zip(bank.names, bank.modules):
            for group, values in ((module.lora_A, state['weights'][name]['A']),
                                  (module.lora_B, state['weights'][name]['B'])):
                for parameter, value in zip(group, values):
                    parameter.copy_(value)
    return bank, state['metadata']
