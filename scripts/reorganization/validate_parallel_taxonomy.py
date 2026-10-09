"""CPU-only audit of the V2a/V2b migration; no model imports or inference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast
import json
import re
import shutil
import tempfile

from validate import sha, targets
from render_gallery import verify

SUB = Path(__file__).resolve().parents[2]
AUDIT = SUB / 'archive/taxonomy_v2a_v2b_20261010'
OLD = SUB / 'archive/reorganization_20261010'


def read(path):
    return json.loads(path.read_text())


def links(path):
    content = path.read_text()
    yield from targets(content)
    if path.suffix == '.html':
        yield from re.findall(r'<option\s+value="([^"]+)"', content)


def main():
    migration = read(AUDIT / 'migration.json')
    mapping = migration['file_path_map']

    def current(path):
        return SUB / mapping.get(path, path)

    # Explicit permission list: current narrative and derived navigation only.
    allowed = set(migration['modified_current_documents'])
    allowed.add('branches/EXPERIMENT_INDEX.json')
    allowed.update(str(p.relative_to(SUB)) for p in (SUB / 'branches').glob('*/*/manifest.json'))
    allowed.update(str(p.relative_to(SUB)) for p in (SUB / 'mainline').glob('*/manifest.json'))
    before = read(AUDIT / 'before_files.json')
    unchanged, changes, protected = [], [], []
    for row in before:
        p = current(row['path'])
        assert p.is_file(), ('missing_pre_migration_file', row['path'], str(p))
        rel = str(p.relative_to(SUB))
        if sha(p) == row['sha256']:
            unchanged.append(rel)
        else:
            assert rel in allowed, ('unexpected_byte_change', row['path'], rel)
            changes.append(rel)
        if (p.suffix in {'.mp4', '.pt', '.safetensors', '.csv', '.log', '.py', '.patch'}
                or '/receipts/' in rel or rel.startswith('code/')):
            assert sha(p) == row['sha256'], ('protected_evidence_changed', rel)
            protected.append(rel)

    # Also retain the original pre-reorganization baseline, not just the rename baseline.
    original = read(OLD / 'before_tracked.json')
    actions = {r['source']: r for r in read(OLD / 'layout_actions.json')}
    for row in original['files']:
        if row['path'] not in actions:
            assert sha(current(row['path'])) == row['sha256'], row['path']
        elif 'original_bytes' in actions[row['path']]:
            assert sha(current(actions[row['path']]['original_bytes'])) == row['sha256']

    for row in migration['archived_overlays']:
        assert sha(SUB / row['new']) == row['sha256'], row
    copies = read(OLD / 'copy_manifest.json')
    for row in copies:
        assert sha(current(row['target'])) == row['sha256'] == sha(SUB.parent / row['source']), row
    for row in read(OLD / 'gallery_copies.json'):
        assert sha(current(row['target'])) == row['sha256'] == sha(current(row['source'])), row
    for row in read(AUDIT / 'gallery_copies.json'):
        assert sha(SUB / row['target']) == row['sha256'] == sha(SUB / row['source']), row
    for row in read(AUDIT / 'selected_sources.json'):
        assert sha(SUB / row['packaged_video']) == row['sha256'] == sha(SUB.parent / row['source'])
    for row in read(OLD / 'link_repairs.json')['imports']:
        assert sha(current(row['target'])) == row['packaged_sha256'], row
        assert sha(SUB.parent / row['source']) == row['sha256'], row
    source_count = 0
    for p in (SUB / 'report').glob('V*/code/SOURCE_MANIFEST.json'):
        for row in read(p):
            assert sha(p.parent / row['filename']) == row['sha256'] == sha(current(row['source']))
            source_count += 1
    frozen = SUB / 'report/V2b_same_sigma_local_bidir/code'
    launch = read(SUB / 'experiments/11_causal_12_then5_selfhistory/launch.json')
    runtime = read(SUB / 'reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/source_runtime_manifest.json')
    for fn in ['run.py', 'interval_forward.py']:
        assert sha(frozen / fn) == launch['sources_sha256'][fn], fn
    for fn in ['local_topology.py', 'h3_cached.py']:
        assert sha(frozen / fn) == runtime[f'code/causal/{fn}'], fn

    # All local Markdown/HTML targets must resolve, including archived narrative.
    docs = [p for p in SUB.rglob('*') if p.suffix in {'.md', '.html'} and '.git' not in p.parts]
    link_count = 0
    broken = []
    for p in docs:
        for url in links(p):
            link_count += 1
            if not (p.parent / url).exists():
                broken.append({'document': str(p.relative_to(SUB)), 'target': url})
    assert not broken, broken

    # Decode every physical packaged MP4; identical bytes share one decode receipt.
    videos = []
    for directory in ['report', 'mainline', 'archive/referenced_assets', 'archive/obsolete_presentation_v2_v3']:
        videos.extend((SUB / directory).rglob('*.mp4'))
    by_hash = {}
    for p in videos:
        by_hash.setdefault(sha(p), []).append(p)

    def check(item):
        digest, paths = item
        result = verify(paths[0])
        assert result['sha256'] == digest
        return dict(files=[str(p.relative_to(SUB)) for p in paths], **result)

    with ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(check, by_hash.items()))
    (AUDIT / 'all_packaged_video_validation.json').write_text(json.dumps(verified, indent=2) + '\n')
    decoded = {r['sha256']: r for r in verified}
    gallery = read(AUDIT / 'current_gallery.json')
    assert len(gallery) == 22
    gallery_by_name = {row['filename']: row for row in gallery}
    for row in gallery:
        p = SUB / 'report/00_comparison_gallery' / row['filename']
        result = decoded[sha(p)]
        assert result['sha256'] == row['video']['sha256']
        assert result['frames'] == row['video']['frames']
        if 'alias_of' in row:
            assert row['video']['sha256'] == gallery_by_name[row['alias_of']]['video']['sha256']
        else:
            assert result['frames'] == row['frames']
        for source in row.get('sources', []):
            assert sha(SUB / source['path']) == source['sha256'], source
    for row in read(AUDIT / 'render_results.json'):
        assert sha(SUB / 'scripts/reorganization/render_parallel_gallery.py') == row['taxonomy_renderer_sha256']

    py = list((SUB / 'scripts/reorganization').glob('*.py')) + list((SUB / 'report').glob('V*/code/*.py'))
    for p in py:
        ast.parse(p.read_text(), filename=str(p))

    # A real isolated copy proves that the report has no external link dependency.
    portable_files = 0
    with tempfile.TemporaryDirectory(prefix='h3-parallel-report-') as td:
        dest = Path(td) / 'report'
        shutil.copytree(SUB / 'report', dest, symlinks=True)
        for p in dest.rglob('*'):
            assert not p.is_symlink(), ('report_symlink', str(p))
            if p.suffix in {'.md', '.html'}:
                for url in links(p):
                    q = (p.parent / url).resolve()
                    assert q.is_relative_to(dest) and q.exists(), (str(p), url)
            if p.is_file():
                assert sha(p) == sha(SUB / 'report' / p.relative_to(dest))
                portable_files += 1

    receipt = dict(
        status='PASS', base_commit=original['commit'],
        original_tracked_files=len(original['files']),
        pre_taxonomy_files=len(before), unchanged_or_byte_preserved_files=len(unchanged),
        intentional_narrative_or_navigation_changes=changes,
        protected_evidence_files=len(protected), missing_files=[], unexpected_changes=[],
        archived_original_overlay_files=len(migration['archived_overlays']),
        original_core_copies_verified=len(copies), source_snapshot_files_verified=source_count,
        V2b_frozen_runtime_hashes_verified=True,
        markdown_html_documents=len(docs), local_links_checked=link_count, broken_links=[],
        current_gallery_videos=len(gallery), newly_encoded_taxonomy_videos=16,
        physical_packaged_MP4=len(videos), unique_decoded_MP4=len(verified),
        full_decode=True, h264_yuv420p=True, constant_24fps_pts=True,
        gallery_frame_counts_match=True, python_ast_files=len(py),
        report_isolated_copy_verified=True, report_has_no_symlinks=True,
        report_files=portable_files,
        report_bytes=sum(p.stat().st_size for p in (SUB / 'report').rglob('*') if p.is_file()),
        no_training_or_model_inference=True,
    )
    (AUDIT / 'validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (AUDIT / 'validation.json').write_text(json.dumps(dict(status='FAIL', error=repr(exc)), ensure_ascii=False, indent=2) + '\n')
        raise
