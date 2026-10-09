"""Final CPU-only integrity, source, Markdown, portable-report and video audit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote
from render_gallery import verify

SUB=Path(__file__).resolve().parents[2]
AUDIT=SUB/'archive/reorganization_20261010'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def targets(text):
    text=re.sub(r'```.*?```','',text,flags=re.S)
    urls=re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text)
    urls+=re.findall(r'(?:href|src)=["\']([^"\']+)["\']',text)
    for u in urls:
        u=unquote(u.strip('<>')).split('#',1)[0].split('?',1)[0]
        if u and not (re.match(r'^[a-zA-Z]+:',u) or u.startswith('//')):yield u


def main():
    # Receipts link to each other. Mark them pending until their checks finish.
    for name in ['validation.json','legacy_link_issues.json','all_packaged_video_validation.json']:
        p=AUDIT/name
        if not p.exists():p.write_text('{"status":"PENDING_VALIDATION"}\n')
    baseline=json.loads((AUDIT/'before_tracked.json').read_text())
    actions=json.loads((AUDIT/'layout_actions.json').read_text())
    allowed={x['source'] for x in actions}
    missing=[];unexpected=[];unchanged=0
    for row in baseline['files']:
        p=SUB/row['path']
        if row['path'] in allowed:continue
        if not p.exists():missing.append(row['path'])
        elif sha(p)!=row['sha256']:unexpected.append(row['path'])
        else:unchanged+=1
    # Every changed/moved old document retains its original bytes in Git or raw archive.
    for a in actions:
        if 'original_bytes' in a:
            old=next(x for x in baseline['files'] if x['path']==a['source'])
            assert sha(SUB/a['original_bytes'])==old['sha256'],a
    assert not missing and not unexpected,(missing,unexpected)
    copies=json.loads((AUDIT/'copy_manifest.json').read_text())
    for r in copies:
        assert sha(SUB/r['target'])==r['sha256']==sha(SUB.parent/r['source']),r
    for r in json.loads((AUDIT/'gallery_copies.json').read_text()):
        assert sha(SUB/r['target'])==r['sha256']==sha(SUB/r['source']),r
    for x in json.loads((AUDIT/'selected_sources.json').read_text()):
        assert sha(SUB.parent/x['source'])==x['sha256']==sha(SUB/x['packaged_video'])
    repairs=json.loads((AUDIT/'link_repairs.json').read_text())
    for r in repairs['imports']:
        assert sha(SUB.parent/r['source'])==r['sha256'],r
        assert sha(SUB/r['target'])==r['packaged_sha256'],r
    for p in (SUB/'report').glob('V*/code/SOURCE_MANIFEST.json'):
        for c in json.loads(p.read_text()):
            assert 'filename' in c,c
            assert sha(p.parent/c['filename'])==c['sha256']==sha(SUB/c['source'])
    # V3 frozen runner and runtime exact hashes, beyond package-snapshot provenance.
    v3=SUB/'report/V3_same_sigma/code'
    launch=json.loads((SUB/'experiments/11_causal_12_then5_selfhistory/launch.json').read_text())
    for fn in ['run.py','interval_forward.py']:assert sha(v3/fn)==launch['sources_sha256'][fn]
    rt=json.loads((SUB/'reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/source_runtime_manifest.json').read_text())
    for fn in ['local_topology.py','h3_cached.py']:assert sha(v3/fn)==rt[f'code/causal/{fn}']
    # New and maintained navigation must resolve; old raw reports remain separate.
    docs=[]
    for base in ['mainline','branches','report']:
        docs+=list((SUB/base).rglob('*.md'))+list((SUB/base).rglob('*.html'))
    docs += [SUB/p for p in ['README.md','REPORT.md','REORGANIZATION_SUMMARY.md',
              'archive/README.md','archive/reorganization_20261010/README.md',
              'archive/historical_docs/README.md','archive/legacy_reports/README.md','docs/archive/README.md']]
    broken=[];count=0
    for p in docs:
        for u in targets(p.read_text()):
            count+=1
            if not (p.parent/u).exists():broken.append(dict(path=str(p.relative_to(SUB)),target=u))
    assert not broken,broken
    # Audit all legacy Markdown without silently editing historical evidence.
    legacy=[]
    for p in SUB.rglob('*.md'):
        if '.git' in p.parts or p in docs:continue
        for u in targets(p.read_text()):
            if not (p.parent/u).exists():legacy.append(dict(path=str(p.relative_to(SUB)),target=u))
    # Compare changed documents' existing links to baseline; don't introduce new failures.
    oldfiles={x['path'] for x in baseline['files']}
    olddirs={str(Path(p).parent) for p in oldfiles}
    for p in list(olddirs):
        olddirs.update(str(x) for x in Path(p).parents)
    def old_exists(path):
        try:r=str(Path(os.path.normpath(path)).relative_to(SUB))
        except ValueError:return Path(path).exists()
        return r in oldfiles or r in olddirs
    moved={a['target']:a['source'] for a in actions if a['operation'].startswith('git mv')}
    new_legacy=[]
    cache={}
    for b in legacy:
        old=moved.get(b['path'],b['path'])
        if old not in oldfiles:
            if b['path']=='archive/legacy_reports/README_before_reorganization.md':old='README.md'
            else:new_legacy.append(b);continue
        if old not in cache:
            txt=subprocess.check_output(['git','show',f'{baseline["commit"]}:{old}'],cwd=SUB,text=True)
            cache[old]=[u for u in targets(txt) if not old_exists((SUB/old).parent/u)]
        # Rebased archive has same unresolved target after normalization.
        now=Path(os.path.normpath((SUB/b['path']).parent/b['target']))
        oldbad=[Path(os.path.normpath((SUB/old).parent/u)) for u in cache[old]]
        if now not in oldbad:new_legacy.append(b)
    (AUDIT/'legacy_link_issues.json').write_text(json.dumps(dict(preexisting_issues=legacy,new_issues=new_legacy),ensure_ascii=False,indent=2)+'\n')
    assert not new_legacy,new_legacy[:20]
    # Deduplicate decode work by SHA, but verify every physical newly packaged MP4.
    videos=(list((SUB/'report').rglob('*.mp4'))+list((SUB/'mainline').rglob('*.mp4'))
            +list((SUB/'archive/referenced_assets').rglob('*.mp4')))
    byhash={}
    for p in videos:byhash.setdefault(sha(p),[]).append(p)
    def check(item):
        h,paths=item;v=verify(paths[0]);assert v['sha256']==h
        return dict(files=[str(p.relative_to(SUB)) for p in paths],**v)
    with ThreadPoolExecutor(max_workers=2) as ex:validated=list(ex.map(check,byhash.items()))
    (AUDIT/'all_packaged_video_validation.json').write_text(json.dumps(validated,indent=2)+'\n')
    # Syntax only, no import/execute of inference code.
    py=list((SUB/'scripts/reorganization').glob('*.py'))+list((SUB/'report').glob('V*/code/*.py'))
    for p in py:ast.parse(p.read_text(),filename=str(p))
    # Actual copy to an isolated directory; all report links must stay within it.
    with tempfile.TemporaryDirectory(prefix='h3-report-portability-') as td:
        dest=Path(td)/'report';shutil.copytree(SUB/'report',dest)
        portable=[]
        for p in dest.rglob('*'):
            assert not p.is_symlink(),p
            if p.suffix in ['.md','.html']:
                for u in targets(p.read_text()):
                    q=(p.parent/u).resolve()
                    assert q.is_relative_to(dest) and q.exists(),(p,u)
            if p.is_file():assert sha(p)==sha(SUB/'report'/p.relative_to(dest))
            portable.append(str(p.relative_to(dest)))
    # Whitespace checks for maintained/new docs and source. Preserve original CSV/log bytes.
    newpaths=[]
    for base in ['mainline','branches','report','scripts/reorganization']:
        newpaths += [p for p in (SUB/base).rglob('*') if p.suffix in ('.md','.py','.json','.html')]
    whitespace=[]
    for p in newpaths:
        for i,l in enumerate(p.read_text().splitlines(),1):
            if l.rstrip()!=l:whitespace.append(f'{p.relative_to(SUB)}:{i}')
    # Historical copied Python may contain whitespace: report it without rewriting source.
    source_whitespace=[x for x in whitespace if '/code/' in x]
    authored_whitespace=[x for x in whitespace if '/code/' not in x and '/receipts/' not in x]
    assert not authored_whitespace,authored_whitespace[:10]
    receipt=dict(status='PASS',base_commit=baseline['commit'],original_tracked_files=len(baseline['files']),
        unchanged_original_files=unchanged,intentionally_changed_documents=len(allowed),
        missing_original_files=missing,unexpected_changes=unexpected,
        original_measurements_checkpoints_videos_and_production_code_unchanged=True,
        copied_sources_verified=len(copies),V3_frozen_source_verified=True,
        maintained_navigation_documents=len(docs),local_links_checked=count,broken_new_links=broken,
        legacy_preexisting_unresolved_links=len(legacy),new_legacy_broken_links=new_legacy,
        legacy_link_repair_occurrences=repairs['repaired_occurrences'],
        recovered_external_assets=len(repairs['imports']),
        physical_packaged_MP4=len(videos),unique_packaged_MP4=len(validated),
        full_video_decode=True,constant_24fps_pts=True,h264_yuv420p=True,
        python_ast_files=len(py),frozen_source_whitespace_preserved=source_whitespace,
        report_actual_isolated_copy_verified=True,portable_entries=len(portable),
        report_bytes=sum(p.stat().st_size for p in (SUB/'report').rglob('*') if p.is_file()),
        no_training_or_model_inference=True)
    (AUDIT/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        (AUDIT/'validation.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc)),ensure_ascii=False,indent=2)+'\n')
        raise
