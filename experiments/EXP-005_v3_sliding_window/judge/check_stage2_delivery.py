"""File-only delivery audit; no model loading or GPU work."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SUB = EXP.parents[1]
ROOT = SUB.parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

errors = []
paths = {ROOT / n for n in ('README.md', 'guideline.md', 'progress.md', 'next_plan.md', 'report.md')}
paths.update(SUB / n for n in ('README.md', 'REPORT.md', 'INTERVIEW_ANSWER.md'))
for folder in ('report', 'mainline'):
    paths.update((SUB / folder).rglob('*.md'))
    paths.update((SUB / folder).rglob('*.html'))
paths.update(EXP.glob('*.md'))
paths.update((EXP / 'stage2').glob('*.md'))
paths.add(HERE / 'STAGE2_FINAL_REVIEW.md')
links = 0
for path in sorted(paths):
    if path.name.startswith(('previous_', 'worker_', 'taskbook_', 'guideline_snapshot')):
        continue
    body = path.read_text()
    targets = re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)', body)
    targets += re.findall(r'(?:href|src|value)=[\"\']([^\"\']+)[\"\']', body)
    for target in targets:
        target = target.strip().split(' "')[0].strip('<>')
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
            continue
        target = unquote(target.split('#')[0].split('?')[0])
        if not target:
            continue
        links += 1
        if not (path.parent / target).exists():
            errors.append({'source': str(path.relative_to(ROOT)), 'target': target})

frozen = []
release = json.loads((SUB / 'mainline/v3/v3_baseline/release_manifest.json').read_text())
for name, digest in release['frozen_code'].items():
    frozen.append((SUB / 'experiments/EXP-003_native_cached_124' / name, digest))
frozen.append((SUB / 'experiments/EXP-003_native_cached_124/config.json', release['config_sha256']))
for name, digest in release['videos'].items():
    frozen.append((SUB / 'report/v3/v3_baseline/videos' / name, digest))
for stage in ('g0', 'g1', 'l1'):
    auth = json.loads((EXP / f'stage2/{stage}_authorization.json').read_text())
    for key, suffix in [('runner_sha256', 'runner.py'), ('config_sha256', 'config.json'), ('manifest_sha256', 'source_manifest.json')]:
        frozen.append((EXP / f'stage2/{stage}_{suffix}', auth[key]))
    manifest = json.loads((EXP / f'stage2/{stage}_source_manifest.json').read_text())
    # Large tensor/checkpoint files were hashed by preflight and at GPU entry.
    # Recheck small source and result dependencies here without rereading tens of GB.
    for row in manifest['sources'].values():
        path = Path(row['path'])
        if path.suffix not in ('.pt', '.npy', '.safetensors', '.bin'):
            frozen.append((path, row['sha256']))
for path, digest in frozen:
    if sha(path) != digest:
        errors.append({'hash_mismatch': str(path)})

artifacts = json.loads((EXP / 'artifact_manifest_stage2.json').read_text())['files']
for row in artifacts:
    if sha(EXP / row['archived']) != row['sha256']:
        errors.append({'artifact_hash_mismatch': row['archived']})
root_md = sorted(p.name for p in ROOT.glob('*.md'))
assert root_md == ['README.md', 'archive.md', 'guideline.md', 'next_plan.md', 'progress.md', 'report.md']
git = ['git', '-c', f'safe.directory={SUB.resolve()}', '-C', str(SUB)]
changed_frozen = subprocess.check_output(git + ['diff', 'bd5711d18b4d35ec8711d52c39c205bd2a96264e', '--name-only', '--', 'experiments/EXP-002_native_cached', 'experiments/EXP-003_native_cached_124', 'experiments/EXP-004_v3_8step', 'mainline/v3/v3_baseline', 'report/v3/v3_baseline'], text=True).splitlines()
if changed_frozen:
    errors.append({'frozen_baseline_changes': changed_frozen})
out = dict(local_links_checked=links, source_and_baseline_hashes_checked=len(frozen), archived_artifact_hashes_checked=len(artifacts), root_markdown_files=root_md, errors=errors)
(HERE / 'stage2_delivery_checks.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(out, ensure_ascii=False))
raise SystemExit(bool(errors))
