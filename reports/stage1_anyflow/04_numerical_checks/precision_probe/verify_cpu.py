"""Actual small H3 + DiffSynth offload wrappers: dtype/gradient/restore audit."""
import copy
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'DiffSynth-Studio-h3-v2')]
import torch
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.anyflow import install_anyflow
from diffsynth.core.vram.layers import AutoWrappedLinear
from precision_profile import precision_profile

torch.set_num_threads(2)
torch.manual_seed(22)
model = make_small_h3().eval().to(torch.bfloat16)
# Exercise the real offload policy, not only plain nn.Linear.
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
def forward(cache, index=0, commit=False, grad=False):
    start = index * 5
    return chunk_forward(model, batch['noise'][:, :, start:start+5], sigma=0 if commit else .6,
        target_sigma=0., index=index, cache=cache, commit=commit, allow_grad_read=grad,
        use_gradient_checkpointing=grad, **common)
with torch.no_grad():
    baseline = forward(H3ChunkCache(2, 'cpu'))
report = []
for profile in ('legacy', 'time_fp32', 'boundary_fp32'):
    model.zero_grad(set_to_none=True)
    with precision_profile(model, profile) as metadata:
        cache = H3ChunkCache(2, 'cpu')
        with torch.no_grad():
            forward(cache, commit=True)
        before = cache.commits
        value = forward(cache, index=1, grad=True)
        value.float().square().mean().backward()
        assert torch.isfinite(value).all() and cache.commits == before
        gradients = [b.attn.qkv_proj.lora_B.grad for b in model.blocks]
        assert all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
        # Exercise a residency cycle while the profile is active.
        model.time_embedder.proj_in.offload()
        model.time_embedder.proj_in.onload()
        assert model.time_embedder.proj_in.computation_dtype == (
            torch.bfloat16 if profile == 'legacy' else torch.float32)
        row = dict(metadata, output_dtype=str(value.dtype), gradient_norms=[float(g.float().norm()) for g in gradients])
    assert all(torch.equal(v, initial[k]) and v.dtype == initial[k].dtype for k, v in model.state_dict().items())
    with torch.no_grad():
        restored = forward(H3ChunkCache(2, 'cpu'))
    assert torch.equal(restored, baseline)
    report.append(dict(row, weights_unchanged=True, legacy_output_restored_bitwise=True))
result = dict(status='passed', scope='small random actual H3, CPU only; not video quality', profiles=report)
(OUT / 'cpu_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
