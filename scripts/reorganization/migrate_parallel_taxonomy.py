"""Apply the user-requested V2a/V2b parallel taxonomy without changing evidence."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

SUB=Path(__file__).resolve().parents[2]
AUDIT=SUB/'archive/taxonomy_v2a_v2b_20261010'
DIRS={
 'mainline/V2_rgb_anchor_causal':'mainline/V2a_rgb_anchor_causal',
 'mainline/V3_same_sigma_history':'mainline/V2b_same_sigma_local_bidir',
 'mainline/V4_efficient_causal_planned':'mainline/V3_efficient_causal_planned',
 'report/V2_rgb_anchor':'report/V2a_rgb_anchor',
 'report/V3_same_sigma':'report/V2b_same_sigma_local_bidir'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    AUDIT.mkdir(parents=True,exist_ok=False)
    tracked=list(filter(None,subprocess.check_output(['git','ls-files','-z'],cwd=SUB,text=True).split('\0')))
    baseline=[dict(path=p,sha256=sha(SUB/p),bytes=(SUB/p).stat().st_size) for p in tracked]
    (AUDIT/'before_files.json').write_text(json.dumps(baseline,indent=2)+'\n')
    mapping={};moves=[]
    def move(old,new):
        src,dst=SUB/old,SUB/new;assert src.exists() and not dst.exists(),(old,new)
        files=[src] if src.is_file() else [p for p in src.rglob('*') if p.is_file()]
        pairs=[(str(p.relative_to(SUB)),new+(str(p)[len(str(src)):] if p!=src else '')) for p in files]
        dst.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','mv','--',old,new],cwd=SUB,check=True)
        for o,n in pairs:
            for key,value in list(mapping.items()):
                if value==o:mapping[key]=n
            if o not in mapping:mapping[o]=n
        moves.append(dict(source=old,target=new))
    for old,new in DIRS.items():move(old,new)
    # Preserve old on-screen labels in a clearly obsolete presentation archive.
    archived=[]
    for base in ['report/00_comparison_gallery','report/V2a_rgb_anchor/videos','report/V2b_same_sigma_local_bidir/videos']:
        for p in sorted((SUB/base).glob('*.mp4')):
            if p.stem.startswith(('V2_','V3_')) and ('_vs_' in p.stem or 'four_paths' in p.stem):
                old=str(p.relative_to(SUB));new='archive/obsolete_presentation_v2_v3/'+old
                move(old,new);archived.append(dict(old=old,new=new,sha256=sha(SUB/new)))
    # Raw source copies contain no numeric version overlay; rename without transcoding.
    for base,oldid,newid in [('report/V2a_rgb_anchor/videos','V2_','V2a_'),('report/V2b_same_sigma_local_bidir/videos','V3_','V2b_')]:
        for p in sorted((SUB/base).glob(oldid+'*.mp4')):move(str(p.relative_to(SUB)),base+'/'+p.name.replace(oldid,newid,1))
    # Current presentation documents get new IDs; original experiments/logs stay historical.
    current=[]
    for base in ['mainline','branches','report']:
        current += [p for p in (SUB/base).rglob('*') if p.suffix in ('.md','.html')]
    current += [SUB/x for x in ['README.md','REPORT.md','REORGANIZATION_SUMMARY.md']]
    # Filename replacements precede numeric label replacements; V2a/b won't be rematched.
    special={'V3_vs_V2':'V2a_vs_V2b', 'V3_A_vs_V2':'V2a_vs_V2b_A','V3_D_vs_V2':'V2a_vs_V2b_D'}
    def convert(text):
        for old,new in DIRS.items():text=text.replace(old,new)
        # Relative paths frequently contain only the leaf directory name.
        for old,new in DIRS.items():text=text.replace(old.split('/')[-1]+'/',new.split('/')[-1]+'/')
        for old,new in special.items():text=text.replace(old,new)
        text=re.sub(r'V([234])(?![0-9ab])',lambda m:{'2':'V2a','3':'V2b','4':'V3'}[m.group(1)],text)
        return text
    modified=[]
    for p in current:
        raw=AUDIT/'previous_documents'/p.relative_to(SUB);raw=raw.with_suffix('.txt');raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(p.read_bytes())
        old=p.read_text();new=convert(old)
        if old!=new:p.write_text(new);modified.append(str(p.relative_to(SUB)))
    # Fix pointers in original maintained docs, without globally renumbering old plans.
    for p in [SUB/'INTERVIEW_ANSWER.md',SUB/'REPRODUCE.md',SUB/'docs/EXPERIMENT_REPORT.md',SUB/'meeting/README.md',
              SUB/'experiments/01_causal_chunk_kv_rollout/README.md',SUB/'experiments/02_rgb_anchor_visual_drift_repair/README.md',
              SUB/'reports/stage1_anyflow/README.md']:
        text=p.read_text();head,sep,body=text.partition('\n\n')
        new=convert(head)+sep+body
        for old,dst in DIRS.items():new=new.replace(old,dst)
        if new!=text:
            raw=AUDIT/'previous_documents'/p.relative_to(SUB);raw=raw.with_suffix('.txt');raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(p.read_bytes())
            p.write_text(new);modified.append(str(p.relative_to(SUB)))
    # Immutable historical receipts retain old asset IDs; fresh manifest records display identity.
    assets=json.loads((SUB/'archive/reorganization_20261010/selected_sources.json').read_text())
    for x in assets:
        x['legacy_asset_id']=x['id']
        x['display_version']={0:'V0',1:'V1',2:'V2a',3:'V2b'}[x['version']]
        x['packaged_video']=mapping.get(x['packaged_video'],x['packaged_video'])
        assert (SUB/x['packaged_video']).exists()
    (AUDIT/'selected_sources.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2)+'\n')
    for d,id_ in [('report/V2a_rgb_anchor','V2a'),('report/V2b_same_sigma_local_bidir','V2b')]:
        p=SUB/d/'PROVENANCE.json';obj=json.loads(p.read_text());obj['historical_version_label']=obj['version'];obj['version']=id_
        obj['current_assets']=[x for x in assets if x['display_version']==id_]
        obj['taxonomy_note']='Parallel research approaches to generated-history conditioning; no checkpoint inheritance between V2a/V2b.'
        p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');modified.append(str(p.relative_to(SUB)))
    receipt=dict(directory_moves=DIRS,file_path_map=mapping,moves=moves,archived_overlays=archived,
                 modified_current_documents=modified,
                 version_ids={'old_V2':'V2a','old_V3':'V2b','old_V4':'V3 planned'},
                 relationship='V1 problem branches into V2a/V2b; V2b restores Original protocol. Future V3 is planned unification.',
                 no_training_or_model_inference=True)
    (AUDIT/'migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(moves=len(moves),archived_MP4=len(archived),modified_documents=len(modified))))


if __name__=='__main__':main()
