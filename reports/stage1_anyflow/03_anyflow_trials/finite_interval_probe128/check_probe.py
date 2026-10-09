"""CPU integration check of the numerical probe with the real tiny H3 path."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('finite_probe', OUT/'probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sys.path[:0] = [str(probe.RUNTIME/'code'), str(probe.RUNTIME/'code/abot'),
               str(probe.RUNTIME/'DiffSynth-Studio-h3-v2')]
import torch
probe.torch = torch
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import chunk_forward
from causal.train_stage1_anyflow import condition, clean_cache
from causal.anyflow import install_anyflow

torch.set_num_threads(2); torch.manual_seed(13)
model = make_small_h3().eval()
install_anyflow(model)
model.requires_grad_(False)
batch = synthetic_h3_batch(frames=12, seed=13)
case = dict(clean=batch['clean_video'], prompt=batch['prompt_embeds'],
    packed=batch['packed'], audio=batch['audio_latents'],
    anchors=[batch['anchor_rows']]*3, actions=None, action_adapter=None)
args = SimpleNamespace(chunk_frames=5, history_chunks=5, anchor_mode='fixed',
                       action_prefix_mode='causal', action_feedback=True)
result = dict(status='running', scope='random small H3, fixed anchor; not pretrained/RGB validation',
              analytic=probe.self_test(), records=[])
versions = [(p, p._version) for p in model.parameters()]
with torch.no_grad():
    for chunk in range(3):
        cache = clean_cache(model, case, chunk, args)
        before = probe.cache_digest(cache); commits = cache.commits
        common = condition(case, chunk, args)
        clean = case['clean'][:, :, chunk*5:(chunk+1)*5]
        noise = batch['noise'][:, :, chunk*5:(chunk+1)*5]
        for index in (0, 4, 7):
            t, r = probe.SIGMAS[index:index+2]
            def velocity(state, sigma, target):
                return chunk_forward(model, state, sigma=sigma, target_sigma=target,
                                     index=chunk, cache=cache, **common)
            record = probe.interval(velocity, (1-t)*clean+t*noise, t, r)
            record.update(chunk=chunk, native_step=index)
            result['records'].append(record)
        assert cache.commits == commits and probe.cache_digest(cache) == before
        cache.clear()
assert all(p._version == v for p, v in versions)
result.update(status='passed', cache_unchanged=True, parameter_versions_unchanged=True)
(OUT/'cpu_h3_integration.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('records','analytic')}))
