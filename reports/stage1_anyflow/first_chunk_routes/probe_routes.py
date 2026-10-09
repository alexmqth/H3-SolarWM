"""Read-only 2x2 attention-edge ablation on the first generated chunk.

There is no history KV in this experiment. The four branches share identical
tokens, weights, SDPA backend, current state and A/D interventions. This is a
local mechanism test, not a trained model or an accepted rollout variant.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
BRIDGE = BASE.parents[1] / '2026-10-08-18/stage1_real_abot_fm'
RT = BRIDGE / 'runtime'
MANIFEST = BRIDGE.parents[2] / 'data/abot_bridge/encoded_manifest.json'
SIGMAS = (.9395404663085938, .6894410400390625, .24078089904785155)
ROLES = ('own_common', 'own_static', 'causal_common', 'causal_static')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classes():
    import torch
    import geometry_primitives as g
    original_class = g.FullPrefixAttention

    class EdgeAttention(original_class):
        def __init__(self, packed, frames, rows, mode, actions=None, adapter=None):
            if frames != 5 or mode not in ROLES or actions is not None or adapter is not None:
                raise ValueError('This ablation supports only first-chunk five-frame frozen roles')
            super().__init__(packed, frames, rows, mode)
            self.packed = packed

        def build_mask(self, device):
            p, n = self.prefix, self.frames * self.rows
            original = original_class(self.packed, self.frames, self.rows, 'original').build_mask(device)
            causal = original_class(self.packed, self.frames, self.rows, 'causal').build_mask(device)
            ann = torch.full((p,), -1, device=device, dtype=torch.long)
            for frame, (lo, hi) in enumerate(self.spans.tolist()):
                ann[lo:hi] = frame
            common = ann < 0
            frame = torch.arange(n, device=device) // self.rows
            mask = original.clone()
            if self.mode.endswith('_static'):
                mask[:p, p:][common] = False
            if self.mode.startswith('causal_'):
                mask[p:, :p] = (ann[None, :] < 0) | (ann[None, :] <= frame[:, None])
            permitted_changes = torch.zeros_like(mask)
            permitted_changes[:p, p:][common] = True
            permitted_changes[p:, :p] = ann[None, :] >= 0
            assert not bool(((mask != original) & ~permitted_changes).any())
            if self.mode == 'own_common':
                assert torch.equal(mask, original)
            if self.mode == 'causal_static':
                assert torch.equal(mask, causal)
            assert bool(mask.diag().all())
            return mask

    return original_class, EdgeAttention


def preflight():
    import torch
    torch.set_num_threads(4)
    _, edge_class = classes()
    prep = json.loads((BRIDGE / 'counterfactual/preparation.json').read_text())
    records = []
    for row in prep['records']:
        if row['chunk'] != 0:
            continue
        assert sha(row['file']) == row['sha256']
        pair = torch.load(row['file'], map_location='cpu', weights_only=True)
        packed = pair['packed']; p = int(packed['action_video_start']); spans = packed['action_text_rows']
        # Row groups have uniform masks. Reachability proves future action
        # text cannot reach current output via these attention edges alone.
        ann = torch.full((p,), -1, dtype=torch.long)
        for frame, (lo, hi) in enumerate(spans.tolist()):
            ann[lo:hi] = frame
        groups = [torch.nonzero(ann < 0).flatten()]
        groups += [torch.arange(int(lo), int(hi)) for lo, hi in spans.tolist()]
        groups += [torch.arange(p + i * 390, p + (i + 1) * 390) for i in range(5)]
        assert all(len(x) for x in groups)
        for role in ROLES:
            mask = edge_class(packed, 5, 390, role).build_mask('cpu')
            graph = torch.tensor([[bool(mask[q[:, None], k[None, :]].any()) for k in groups] for q in groups])
            reachable = graph.clone()
            for _ in range(50):
                nxt = reachable | (reachable.int() @ graph.int() > 0)
                if torch.equal(nxt, reachable):
                    break
                reachable = nxt
            video_groups = slice(1 + len(spans), None)
            future_actions = slice(1 + 5, 1 + len(spans))
            assert not bool(reachable[video_groups, future_actions].any())
            records.append(dict(clip=row['clip'], role=role, pair_sha256=row['sha256'],
                mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                no_future_action_path_in_mask_graph=True, only_two_declared_edge_classes_changed=True))
    result = dict(status='passed', scope='CPU actual-layout mask identities and graph reachability only',
                  records=records, source_sha256=sha(__file__))
    (BASE / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def run(args):
    import torch
    import infer as abot
    import geometry_primitives as g
    from causal.h3_precision import configure_precision
    from causal.pretrained_lora import load_adapter
    from causal.stage1_lora import load_stage1_lora
    from causal.h3_cached import H3ChunkCache, chunk_forward
    torch.set_num_threads(4); torch.manual_seed(13)
    assert json.loads((BASE / 'preflight.json').read_text())['source_sha256'] == sha(__file__)
    out = BASE / 'probe.json'
    if out.exists():
        raise FileExistsError('Existing GPU receipt; no blind retry')
    result = dict(status='loading', gpu=args.gpu, pid=os.getpid(),
        started_at=datetime.now().astimezone().isoformat(), optimizer_steps=0,
        history_frames=0, state_source='fixed step00-generated first-chunk endpoint/noise interpolants',
        roles=list(ROLES), records=[], model_forwards=0, scope=__doc__,
        source_sha256={str(p): sha(p) for p in [Path(__file__), BRIDGE / 'geometry_primitives.py',
            RT / 'code/causal/h3_cached.py', MANIFEST]})
    began = time.perf_counter()
    def save():
        result['wall_seconds'] = time.perf_counter() - began
        tmp = out.with_suffix('.tmp.json'); tmp.write_text(json.dumps(result, indent=2) + '\n'); tmp.replace(out)
    save()
    original_class, edge_class = classes()
    try:
        pipe = abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(RT / 'checkpoints/H3-World/step-10000.safetensors'), hotload=True)
        model = pipe.dit.requires_grad_(False).eval()
        checkpoint = BRIDGE / 'train_48/step_00'
        trainer = torch.load(checkpoint / 'trainer_state.pt', map_location='cpu', weights_only=True)
        assert trainer['optimizer_step'] == 0
        assert all(sha(checkpoint / n) == h for n, h in trainer['weight_sha256'].items())
        result['checkpoint_sha256'] = trainer['weight_sha256']; del trainer
        load_adapter(model, checkpoint / 'causal_adapter.pt', 'cuda:0')
        configure_precision(model, 'h3_fp32', native_transformer_dir=RT / 'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        load_stage1_lora(model, checkpoint / 'stage1_lora.pt', device='cuda:0')
        pipe.load_models_to_device(['dit']); model.requires_grad_(False)
        versions = [(p, p._version) for p in model.parameters()]
        def move(value):
            if torch.is_tensor(value): return value.to('cuda:0')
            if isinstance(value, dict): return {k: move(v) for k, v in value.items()}
            return value
        manifest = json.loads(MANIFEST.read_text())
        prep = json.loads((BRIDGE / 'counterfactual/preparation.json').read_text())
        baseline = json.loads((BRIDGE / 'geometry/step_00_step00_generated/probe.json').read_text())
        result['status'] = 'running'; save()
        with torch.no_grad():
            for paired in [x for x in prep['records'] if x['chunk'] == 0]:
                row = next(x for x in manifest['clips'] if x['clip_id'] == paired['clip'])
                assert row['split'] == 'validation' and sha(row['encoded_file']) == row['sha256']
                assert sha(paired['file']) == paired['sha256']
                raw = torch.load(row['encoded_file'], map_location='cpu', weights_only=True)
                pair = move(torch.load(paired['file'], map_location='cpu', weights_only=True))
                cond = move(raw['conditioning']); anchor = move(raw['anchors'][0])
                generated_path = BRIDGE / 'eval/step_00/generated_30' / paired['clip'] / 'latents.pt'
                endpoint = torch.load(generated_path, map_location='cpu', weights_only=True).float().to('cuda:0')
                history = endpoint[:, :, :0]; noise = cond['initial_noise'].float()
                for sigma in SIGMAS:
                    state = (1 - sigma) * endpoint[:, :, :5] + sigma * noise[:, :, :5]
                    source = next(r for r in baseline['records'] if r['clip'] == paired['clip'] and r['chunk'] == 0 and r['sigma'] == sigma)
                    assert g.tensor_sha(state) == source['state_sha256']
                    outputs = {}
                    g.FullPrefixAttention = edge_class
                    for role in ROLES:
                        outputs[role] = [g.forward(model, state, history, cond, pair['prompts'][a], anchor,
                            pair['packed'], sigma, mode=role).float().cpu() for a in 'AD']
                        result['model_forwards'] += 2
                    teacher = outputs['own_common']
                    record = dict(clip=paired['clip'], sigma=sigma, state_sha256=g.tensor_sha(state),
                        history_sha256=g.tensor_sha(history), anchor_sha256=g.tensor_sha(anchor),
                        pair_sha256=paired['sha256'], generated_file_sha256=sha(generated_path),
                        teacher_prediction_sha256={a: g.tensor_sha(t) for a, t in zip('AD', teacher)},
                        roles={role: dict(absolute={a: g.compare(v, teacher[i]) for i, (a, v) in enumerate(zip('AD', outputs[role]))},
                            action_delta=g.delta_metrics(outputs[role], teacher)) for role in ROLES})
                    if sigma == SIGMAS[1]:
                        g.FullPrefixAttention = original_class
                        repeat = [g.forward(model, state, history, cond, pair['prompts'][a], anchor,
                            pair['packed'], sigma, mode='original').cpu() for a in 'AD']
                        result['model_forwards'] += 2
                        record['original_identity_rmse'] = {a: g.rms(repeat[i] - teacher[i]) for i, a in enumerate('AD')}
                        assert all(x == 0 for x in record['original_identity_rmse'].values())
                        cache = H3ChunkCache(5, 'cpu')
                        cached = [chunk_forward(model, state, index=0, cache=cache, sigma=sigma,
                            full_packed=pair['packed'], prompt=pair['prompts'][a], anchor=anchor,
                            audio=cond['audio_noise'], chunk_frames=5, anchor_slot=1,
                            action_prefix_mode='causal', action_feedback=True).float().cpu() for a in 'AD']
                        result['model_forwards'] += 2
                        assert cache.commits == 0 and cache.nbytes == 0
                        record['cached_vs_full_causal'] = dict(
                            absolute={a: g.compare(cached[i], outputs['causal_static'][i]) for i, a in enumerate('AD')},
                            action_delta=g.delta_metrics(cached, outputs['causal_static']),
                            caveat='Identical mask predicates; different one-vs-two SDPA query shapes can affect BF16 rounding')
                    assert all(p._version == v for p, v in versions)
                    assert g.tensor_sha(state) == source['state_sha256']
                    result['records'].append(record); save()
                    print(json.dumps(dict(clip=paired['clip'], sigma=sigma,
                        cosines={r: record['roles'][r]['action_delta']['cosine'] for r in ROLES})), flush=True)
        result.update(status='complete', gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated() / 2**20,
            parameter_versions_unchanged=True,
            limitation='First chunk only, no history KV. Local edge intervention does not prove multi-chunk correctness or video quality. All branches use full-prefix SDPA; cached deployment shape control reported separately.')
    except BaseException as exc:
        result.update(status='failed', error=repr(exc)); raise
    finally:
        g.FullPrefixAttention = original_class
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if not args.preflight:
        if args.gpu is None: parser.error('--gpu is required for the read-only H3 run')
        free = int(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        if free < 34000: raise RuntimeError(f'GPU{args.gpu} has only {free}MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.preflight else str(args.gpu),
        ABOT_VRAM_RESERVE_GIB='18', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0] = [str(BRIDGE), str(RT / 'code'), str(RT / 'code/abot'), str(RT / 'DiffSynth-Studio-h3-v2')]
    preflight() if args.preflight else run(args)
