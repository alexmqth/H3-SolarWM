#!/usr/bin/env python3
"""Rebuild the pinned source and validate explicitly supplied external weights.

Never downloads model weights. Only the upstream Git source is cloned.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ['diffsynth_h3_action.patch', 'diffsynth_causal.patch',
           'diffsynth_long_video_mask.patch', 'diffsynth_anyflow.patch',
           'diffsynth_native_fp32.patch']


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minimax-model-dir', required=True, type=Path,
                   help='Existing MiniMax-H3 directory containing FL2VA/')
    p.add_argument('--h3-checkpoint', required=True, type=Path)
    p.add_argument('--out', type=Path, default=ROOT/'outputs/runtime_preflight.json')
    args = p.parse_args()
    base = args.minimax_model_dir.resolve(strict=True)
    h3 = args.h3_checkpoint.resolve(strict=True)
    if not (base/'FL2VA').is_dir() or not h3.is_file():
        p.error('Expected MiniMax-H3/FL2VA and the released H3-World safetensors file')
    checkout = ROOT/'DiffSynth-Studio-h3-v2'
    revision = (ROOT/'code/diffsynth_base_commit.txt').read_text().strip()
    if not checkout.exists():
        run('git', 'clone', '--filter=blob:none', '--no-checkout',
            'https://github.com/modelscope/DiffSynth-Studio.git', str(checkout))
        run('git', '-C', str(checkout), 'checkout', '--detach', revision)
    head = run('git', '-C', str(checkout), 'rev-parse', 'HEAD', capture_output=True).stdout.strip()
    if head != revision:
        raise RuntimeError(f'Expected DiffSynth {revision}, found {head}; use a fresh checkout')
    # Receipt makes reruns verify the exact already-patched tree instead of
    # reverse-applying overlapping patches individually.
    marker = checkout/'.submission_patches.json'
    patch_hashes = {name: sha(ROOT/'code'/name) for name in PATCHES}
    if marker.exists():
        receipt = json.loads(marker.read_text())
        assert receipt['patches'] == patch_hashes, 'Patch set changed: use a fresh checkout'
        for name, expected in receipt['source_files'].items():
            assert sha(checkout/name) == expected, f'Patched source changed: {name}'
    else:
        status = run('git', '-C', str(checkout), 'status', '--porcelain', capture_output=True).stdout
        if status.strip():
            raise RuntimeError('Use a clean DiffSynth checkout before applying the patch stack')
        for name in PATCHES:
            run('git', '-C', str(checkout), 'apply', '--check', str(ROOT/'code'/name))
            run('git', '-C', str(checkout), 'apply', str(ROOT/'code'/name))
        files = list((checkout/'diffsynth').rglob('*.py'))
        receipt = {'base_revision': revision, 'patches': patch_hashes,
                   'source_files': {str(f.relative_to(checkout)): sha(f) for f in files}}
        marker.write_text(json.dumps(receipt, indent=2)+'\n')
    target = checkout/'models/MiniMax/MiniMax-H3'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        assert target.resolve() == base, f'Model path points somewhere else: {target}'
    else:
        target.symlink_to(base, target_is_directory=True)
    from safetensors import safe_open
    dependencies = []
    for component in ['text_encoder', 'transformer']:
        folder = base/'FL2VA'/component
        index = json.loads((folder/'model.safetensors.index.json').read_text())
        shard_keys = {}
        for key, name in index['weight_map'].items():
            shard_keys.setdefault(name, set()).add(key)
        for name, expected_keys in sorted(shard_keys.items()):
            path = folder/name
            with safe_open(str(path), framework='pt', device='cpu') as f:
                assert expected_keys <= set(f.keys()), f'Incomplete shard: {path}'
            dependencies.append({'path': str(path), 'bytes': path.stat().st_size,
                                 'index_keys': len(expected_keys)})
    for name in ['video_vae/source/model.safetensors', 'audio_vae/model.safetensors']:
        path = base/'FL2VA'/name
        with safe_open(str(path), framework='pt', device='cpu') as f:
            count = len(f.keys())
        assert count > 0
        dependencies.append({'path': str(path), 'bytes': path.stat().st_size, 'keys': count})
    assert (base/'FL2VA/processor').is_dir(), 'Missing processor directory'
    with safe_open(str(h3), framework='pt', device='cpu') as f:
        keys = list(f.keys())
    assert keys and all('.lora_A.' in k or '.lora_B.' in k for k in keys)
    assets = ['checkpoints/visual_rgb_tail16/causal_adapter.pt',
              'checkpoints/visual_rgb_tail16/action_adapter.pt', 'examples/first_frame.png']
    adapter_hashes = {name: sha(ROOT/name) for name in assets}
    manifest = json.loads((ROOT/'meeting/DEMO_PROVENANCE.json').read_text())
    assert sha(h3) == manifest['released_h3']['sha256'], 'Released H3 LoRA version mismatch'
    assert sum('.lora_A.' in k for k in keys) == manifest['released_h3']['lora_pairs']
    for name in assets:
        assert manifest['assets'][name]['sha256'] == adapter_hashes[name], name
    result = {'status': 'pass', 'source': receipt, 'weights': dependencies,
              'weight_check': 'Index/shard keys and safetensors headers; not a full hash of base weights',
              'released_h3': {'path': str(h3), 'sha256': sha(h3), 'lora_pairs': sum('.lora_A.' in k for k in keys)},
              'packaged_inputs': adapter_hashes, 'auto_model_download': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source','weights')}, indent=2))


if __name__ == '__main__':
    main()
