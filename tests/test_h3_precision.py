from pathlib import Path
import sys
import copy

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'DiffSynth-Studio-h3-v2')]
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.pretrained_lora import install_adapters
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.h3_precision import OWNERS, configure_precision, validate_precision_checkpoint
from causal.anyflow import install_anyflow, anyflow_sample_loss
from diffsynth.core.vram.layers import AutoWrappedLinear


def model_with_native_checkpoint(path):
    torch.manual_seed(43)
    model = make_small_h3().eval()
    weights = {f'{name}.{slot}': getattr(model.get_submodule(name), slot).detach().clone()
               for name in OWNERS for slot in ('weight', 'bias')}
    path.mkdir()
    save_file(weights, str(path / 'model.safetensors'))
    model.bfloat16()
    for name in OWNERS:
        owner, _, slot = name.rpartition('.')
        parent = model.get_submodule(owner) if owner else model
        layer = AutoWrappedLinear(getattr(parent, slot), computation_dtype=torch.bfloat16,
                                  computation_device='cpu', offload_device='cpu')
        layer.onload()
        setattr(parent, slot, layer)
    for b in model.blocks:
        b.attn.qkv_proj = b.attn.qkv_proj.base
    adapters, _ = install_adapters(model, rank=4, block_indices=[0,1], device='cpu')
    return model, weights, adapters


def test_native_restore_and_offload_preserve_actual_fp32_values(tmp_path):
    model, weights, _ = model_with_native_checkpoint(tmp_path / 'native')
    before = model.time_embedder.proj_in.weight.float().clone()
    assert not torch.equal(before, weights['time_embedder.proj_in.weight'])
    report = configure_precision(model, 'h3_fp32', native_transformer_dir=tmp_path / 'native')
    assert report['native_fp32_weights_restored'] and len(report['native_tensor_sha256']) == 12
    for name in OWNERS:
        layer = model.get_submodule(name)
        layer.offload(); layer.onload()
        assert layer.computation_dtype == layer.weight.dtype == torch.float32
        for slot in ('weight', 'bias'):
            assert torch.equal(getattr(layer, slot), weights[f'{name}.{slot}'])
    delta = install_anyflow(model).delta
    assert torch.equal(delta.proj_in.weight, weights['time_embedder.proj_in.weight'])
    assert torch.equal(delta.proj_out.weight, weights['time_embedder.proj_out.weight'])
    with pytest.raises(ValueError, match='already installed'):
        configure_precision(model, 'legacy')


@pytest.mark.parametrize('anyflow', [False, True])
def test_bf16_stack_fp32_inputs_grad_cache_and_clean_commit(tmp_path, anyflow):
    torch.set_num_threads(2)
    model, weights, adapters = model_with_native_checkpoint(tmp_path / 'native')
    configure_precision(model, 'h3_fp32', native_transformer_dir=tmp_path / 'native')
    if anyflow:
        install_anyflow(model).requires_grad_(False)
    batch = synthetic_h3_batch(frames=7)
    common = dict(full_packed=batch['packed'], prompt=batch['prompt_embeds'].bfloat16(),
        anchor=batch['anchor_rows'].bfloat16(), audio=batch['audio_latents'].bfloat16(), chunk_frames=5)
    seen = []
    handle = model.video_patch_proj.register_forward_pre_hook(lambda _, a: seen.append(a[0].detach().clone()))
    cache = H3ChunkCache(2, 'cpu')
    with torch.no_grad():
        value = chunk_forward(model, torch.ones_like(batch['noise'][:,:,:5])+1e-4,
            sigma=.6, target_sigma=0. if anyflow else None, index=0, cache=cache, **common)
    handle.remove()
    assert sum(int((x == torch.tensor(1.+1e-4)).sum()) for x in seen) == batch['noise'][:,:,:5].numel()
    with torch.no_grad():
        chunk_forward(model, batch['clean_video'][:,:,:5], sigma=0., index=0, cache=cache,
                      commit=True, **common)
    commits = cache.commits
    prediction = chunk_forward(model, batch['noise'][:,:,5:], sigma=.6,
        target_sigma=0. if anyflow else None, index=1, cache=cache,
        allow_grad_read=True, use_gradient_checkpointing=True, **common)
    prediction.square().mean().backward()
    assert value.dtype == prediction.dtype == torch.float32
    assert cache.commits == commits
    assert all(a.lora_B.grad is not None and a.lora_B.grad.norm()>0 and torch.isfinite(a.lora_B.grad).all() for a in adapters)
    assert all(torch.equal(model.get_submodule(n).weight, weights[f'{n}.weight']) for n in OWNERS)


def test_anyflow_preserves_fp32_noisy_and_perturbed_callback_inputs():
    clean, noise = torch.randn(1,2,3).bfloat16(), torch.randn(1,2,3).bfloat16()
    p = torch.nn.Parameter(torch.tensor(.37))
    seen = []
    def fn(x,t,r):
        seen.append(x.detach().clone())
        return torch.tanh(p*x + .11*t + .19*r)
    loss, weight, _ = anyflow_sample_loss(fn, clean, noise, .713, .23, preserve_fp32_inputs=True)
    gradient, = torch.autograd.grad(loss*weight, p)
    assert len(seen)==4 and all(v.dtype==torch.float32 for v in seen)
    assert any(not torch.equal(v, v.bfloat16().float()) for v in seen)
    oracle, ow, _ = anyflow_sample_loss(fn, clean.float(), noise.float(), .713, .23)
    og, = torch.autograd.grad(oracle*ow, p)
    torch.testing.assert_close(loss, oracle, atol=0, rtol=0)
    torch.testing.assert_close(gradient, og, atol=0, rtol=0)


def test_precision_checkpoint_rejects_policy_or_native_weight_mismatch():
    report = dict(profile='h3_fp32', native_tensor_sha256={'a':'hash'})
    meta = dict(config={'precision_profile':'h3_fp32'}, precision=copy.deepcopy(report))
    validate_precision_checkpoint(meta, report)
    with pytest.raises(ValueError, match='precision mismatch'):
        validate_precision_checkpoint(meta, {'profile':'legacy'})
    with pytest.raises(ValueError, match='provenance'):
        validate_precision_checkpoint(meta, dict(report, native_tensor_sha256={'a':'other'}))
    with pytest.raises(ValueError, match='precision mismatch'):
        validate_precision_checkpoint({'config':{}}, report)
    validate_precision_checkpoint({'config':{}}, {'profile':'legacy'})
