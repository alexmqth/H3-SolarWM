import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'DiffSynth-Studio-h3-v2')]
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.anyflow import install_anyflow
from causal.h3_precision import configure_precision
from causal.pretrained_lora import install_adapters
from causal.stage1_lora import install_stage1_lora, save_stage1_lora, load_stage1_lora, target_names
from diffsynth.core.vram.layers import AutoWrappedLinear


@pytest.mark.parametrize('anyflow', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_full_scope_preserves_initial_video_rng_frozen_base_and_roundtrip(tmp_path, dtype, anyflow):
    torch.set_num_threads(2)
    torch.manual_seed(81)
    model = make_small_h3().eval().to(dtype).requires_grad_(False)
    for block in model.blocks:
        block.attn.qkv_proj = block.attn.qkv_proj.base
    # Exercise real offload wrappers beneath a nonzero pretrained visual LoRA.
    for name in target_names(model):
        parent, _, slot = name.rpartition('.')
        wrapper = AutoWrappedLinear(model.get_submodule(name), computation_dtype=dtype,
            computation_device='cpu', offload_device='cpu')
        wrapper.onload()
        setattr(model.get_submodule(parent), slot, wrapper)
    visual, _ = install_adapters(model, rank=4, block_indices=list(range(len(model.blocks))))
    with torch.no_grad():
        for adapter in visual:
            adapter.lora_B.normal_(std=.02)
    model.requires_grad_(False)
    configure_precision(model, 'h3_fp32')
    if anyflow:
        install_anyflow(model).requires_grad_(False)
    fresh = copy.deepcopy(model)
    original = {id(p): p.detach().clone() for p in model.parameters()}
    b = synthetic_h3_batch(frames=7, seed=99)
    common = dict(full_packed=b['packed'], prompt=b['prompt_embeds'].to(dtype),
        anchor=b['anchor_rows'].to(dtype), audio=b['audio_latents'].to(dtype), chunk_frames=5)
    def forward(model, cache, *, grad=False):
        return chunk_forward(model, b['noise'][:,:,5:], sigma=.7, target_sigma=.2 if anyflow else None,
            index=1, cache=cache, allow_grad_read=grad,
            use_gradient_checkpointing=grad, **common)
    def history(model):
        cache = H3ChunkCache(2, 'cpu')
        with torch.no_grad():
            chunk_forward(model, b['clean_video'][:,:,:5], sigma=0., index=0,
                          cache=cache, commit=True, **common)
        return cache
    with torch.no_grad():
        before = forward(model, history(model))
    rng = torch.get_rng_state().clone()
    bank = install_stage1_lora(model, rank=4, alpha=4.)
    for module in bank.modules:
        base = module.base.base if hasattr(module.base, 'base') else module.base
        base.offload(); base.onload()
    assert torch.equal(rng, torch.get_rng_state())
    assert bank.describe()['logical_linear_targets'] == 6 * (len(model.blocks) + len(model.token_refiner.blocks))
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in bank.parameters()}
    cache = history(model)
    with torch.no_grad():
        assert torch.equal(before, forward(model, cache))
    commits = cache.commits
    optimizer = torch.optim.AdamW(bank.parameters(), lr=1e-3)
    forward(model, cache, grad=True).square().mean().backward()
    assert cache.commits == commits
    for group in ('qkv', 'out', 'ffn', 'refiner'):
        assert any(p.grad is not None and p.grad.norm()>0 for p in bank.parameters(group))
        assert all(torch.isfinite(p.grad).all() for p in bank.parameters(group) if p.grad is not None)
    optimizer.step()
    assert all(torch.equal(p, original[id(p)]) and p.grad is None
               for p in model.parameters() if id(p) in original)
    path = tmp_path/'adapter.pt'
    save_stage1_lora(path, bank, {'optimizer_step':1})
    loaded, metadata = load_stage1_lora(fresh, path)
    assert metadata['optimizer_step']==1
    assert all(torch.equal(x,y) for x,y in zip(bank.parameters(),loaded.parameters()))
    with torch.no_grad():
        assert torch.equal(forward(model, history(model)), forward(fresh, history(fresh)))
    with pytest.raises(ValueError, match='already installed'):
        install_stage1_lora(model)


def test_wrong_topology_rejected_before_modification(tmp_path):
    model = make_small_h3().eval().requires_grad_(False)
    bank = install_stage1_lora(model, rank=2, alpha=2.)
    path = tmp_path/'bank.pt';save_stage1_lora(path, bank, {})
    state = torch.load(path, weights_only=True)
    state['targets'].pop();torch.save(state,path)
    fresh = make_small_h3().eval().requires_grad_(False)
    with pytest.raises(ValueError, match='topology'):
        load_stage1_lora(fresh,path)
    assert not hasattr(fresh.blocks[0].mlp.fc1,'lora_A')
