"""Check that tiny trajectory perturbations survive, with actual H3/offload."""
from pathlib import Path
import json
import sys
import torch

OUT = Path(__file__).resolve().parent
RUNTIME = OUT / 'runtime'
sys.path[:0] = [str(RUNTIME / 'code'), str(RUNTIME / 'DiffSynth-Studio-h3-v2')]
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.anyflow import install_anyflow
from diffsynth.core.vram.layers import AutoWrappedLinear
import diffsynth.models.minimax_h3_dit as dit_module
from precision_profile import precision_profile

assert str(RUNTIME) in dit_module.__file__
torch.set_num_threads(2)
torch.manual_seed(22)
model = make_small_h3().eval().bfloat16()
for owner, names in [(model.time_embedder, ('proj_in', 'proj_out')),
                     (model, ('video_patch_proj', 'audio_patch_proj')),
                     (model.final_layer, ('video_out', 'audio_out'))]:
    for name in names:
        wrapped = AutoWrappedLinear(getattr(owner, name), computation_dtype=torch.bfloat16,
                                    computation_device='cpu', offload_device='cpu')
        wrapped.onload()
        setattr(owner, name, wrapped)
install_anyflow(model).requires_grad_(False)
batch = synthetic_h3_batch(frames=7)
common = dict(full_packed=batch['packed'], prompt=batch['prompt_embeds'].bfloat16(),
    anchor=batch['anchor_rows'].bfloat16(), audio=batch['audio_latents'].bfloat16(), chunk_frames=5)
initial = {k: v.clone() for k, v in model.state_dict().items()}

def forward(x, cache, index=0, commit=False, grad=False):
    return chunk_forward(model, x, sigma=0. if commit else .6, target_sigma=0.,
        index=index, cache=cache, commit=commit, allow_grad_read=grad,
        use_gradient_checkpointing=grad, **common)

current = torch.ones_like(batch['noise'][:, :, :5]) + 1e-4
with torch.no_grad():
    baseline = forward(current, H3ChunkCache(2, 'cpu'))
report = []
for profile in ('legacy', 'boundary_fp32', 'boundary_fp32_inputs'):
    model.zero_grad(set_to_none=True)
    captured = []
    with precision_profile(model, profile) as metadata:
        # Runs after the profile's projection input hook. Both boundary
        # profiles input FP32 to the matrix, but only the new one retains
        # the original small perturbation before arriving at that hook.
        h = model.video_patch_proj.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
        cache = H3ChunkCache(2, 'cpu')
        with torch.no_grad():
            value = forward(current, cache)
        h.remove()
        maximum = float(captured[0].max())
        # Anchor rows can be >1. Use the numerous current video rows.
        retained = int((captured[0].float() == current.flatten()[0]).sum())
        if profile == 'boundary_fp32_inputs':
            assert retained == current.numel(), (retained, current.numel())
        else:
            assert retained == 0
        with torch.no_grad():
            forward(batch['clean_video'][:, :, :5], cache, commit=True)
        commits = cache.commits
        prediction = forward(batch['noise'][:, :, 5:], cache, index=1, grad=True)
        prediction.float().square().mean().backward()
        gradients = [b.attn.qkv_proj.lora_B.grad for b in model.blocks]
        assert all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
        assert cache.commits == commits and torch.isfinite(prediction).all()
    assert not hasattr(model, '_diagnostic_input_dtype')
    assert all(v.dtype == initial[k].dtype and torch.equal(v, initial[k]) for k,v in model.state_dict().items())
    with torch.no_grad():
        restored = forward(current, H3ChunkCache(2, 'cpu'))
    assert torch.equal(restored, baseline)
    report.append(dict(metadata, exact_perturbed_video_elements_retained=retained,
        expected_video_elements=current.numel(), finite_nonzero_qkv_gradient=True,
        cache_read_only=True, rollback_bitwise=True, output_dtype=str(prediction.dtype)))
result = dict(status='passed', scope='CPU small H3 diagnostic, not pretrained quality', profiles=report)
(OUT / 'cpu_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
