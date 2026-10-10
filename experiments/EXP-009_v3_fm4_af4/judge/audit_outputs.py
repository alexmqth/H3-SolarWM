"""Independent CPU provenance/output audit; visual judgement stays separate."""
from pathlib import Path
import argparse
import hashlib
import json
import av
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parents[2]
OUT=ROOT/'H3-World/outputs/EXP-009_v3_fm4_af4'
FM8=ROOT/'H3-World/outputs/EXP-006_v3_fm8_full'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

parser=argparse.ArgumentParser()
parser.add_argument('--through',choices=['c2','c3'],required=True)
args=parser.parse_args()
complete=args.through=='c3'
ledger=json.loads((OUT/'budget.json').read_text())
assert 'active_start' not in ledger
expected=(38,32,6,8) if complete else (22,16,6,4)
assert tuple(ledger[k] for k in ('forwards','sampling','commits','vae'))==expected
assert ledger['gpu_seconds']<=.6*3600
assert len(ledger['runs'])==(10 if complete else 6)
assert all(r['status']=='complete_pending_visual_review' for r in ledger['runs'])
mp=EXP/'source_manifest.json'; manifest=json.loads(mp.read_text()); mh=sha(mp)
assert manifest['checkpoint_step']==32
for name,item in manifest['sources'].items():assert sha(ROOT/item['path'])==item['sha256'],name
rows={}; cache_hashes={}
for model in ('fm4','af4'):
    stages=[(None,'prefill',0,12,0)]+[(b,s,a,z,i) for b in ('AA','AD')
        for s,a,z,i in [('second',12,17,1)]+([('third',17,22,2)] if complete else [])]
    for branch,stage,start,stop,index in stages:
        folder=OUT/model if branch is None else OUT/model/branch
        row=json.loads((folder/f'{stage}_{start}_{stop}.json').read_text())
        label=f'{model}/'+('C1' if index==0 else f'{branch}/C{index+1}')
        assert row['status']=='complete_pending_visual_review'
        assert row['model']==model and row['index']==index and row['interval']==[start,stop]
        assert row['checkpoint_step']==(32 if model=='af4' else None)
        assert row['checkpoint_manifest_sha256']==mh
        assert row['runner_sha256']==manifest['sources']['runner']['sha256']
        assert tuple(row[k] for k in ('sampling_forwards','commit_forwards','vae_calls'))==((0,1,0) if index==0 else (4,int(index==1),1))
        assert row['peak_allocated_gib']<=44
        assert row['fm8_first_sha256']==manifest['sources']['fm8_first']['sha256']
        if index<2:
            cache=folder/('cache_through12.pt' if index==0 else 'cache_through17.pt')
            cache_hashes[label]=sha(cache)
            assert cache_hashes[label]==row['cache_after_sha256']
            assert row['cache_after']['entries']==[[0,4680]]+([[1,1950]] if index else [])
            assert row['cache_after']['layer_count']==50
        if index:
            ref=json.loads((FM8/branch/f'chunk_{start}_{stop}.json').read_text())
            assert row['noise_sha256']==ref['current_noise_sha256']
            assert row['prompt_sha256']==ref['prompt_sha256']
            assert row['sigmas']==ref['sigmas'][::2]
            assert row['history_cache_read_unchanged'] and row['source_cache_file_unchanged'] and row['old_RGB_unchanged']
            assert row['cache_before']['entries']==[[0,4680]]+([[1,1950]] if index==2 else [])
            assert row['cache_before']['layer_count']==50
            parent=f'{model}/'+('C1' if index==1 else f'{branch}/C2')
            assert row['source_cache_sha256']==cache_hashes[parent]
            if index==1:assert row['history_tensor_sha256']==ref['history_tensor_sha256']
            assert sha(folder/f'chunk_{start}_{stop}.pt')==row['endpoint_sha256']
            n=39+17*index; rp=folder/f'published_{n}.npy'
            assert sha(rp)==row['published_file_sha256']
            rgb=np.load(rp);assert rgb.shape==(n,480,832,3)
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['published_RGB_sha256']
            prev=np.load(FM8/'published_39.npy' if index==1 else folder/'published_56.npy')
            assert np.array_equal(rgb[:len(prev)],prev)
            video=folder/f'rollout_{n}.mp4';assert sha(video)==row['video_sha256']
            with av.open(str(video)) as c:
                assert c.streams.video[0].average_rate==24
                frames=list(c.decode(video=0)); pts=[f.pts for f in frames]
                assert len(frames)==n and all(b>a for a,b in zip(pts,pts[1:]))
                assert all((f.width,f.height)==(832,480) for f in frames)
        rows[label]=row
assert cache_hashes['fm4/C1']!=cache_hashes['af4/C1']
result=dict(task='EXP-009/v1-FM4-AF4',through=args.through,independent_execution_audit='PASS',
    visual_assessment='SEPARATE',forwards=ledger['forwards'],sampling=ledger['sampling'],
    commits=ledger['commits'],vae=ledger['vae'],updates=0,gpu_seconds=ledger['gpu_seconds'],
    gpu_hours=ledger['gpu_seconds']/3600,max_allocated_gib=max(r['peak_allocated_gib'] for r in rows.values()),
    rows={n:{k:r.get(k) for k in ('sampling_seconds','commit_seconds','decode_seconds','seconds','flow',
        'boundary_gray_MAD','inside_gray_MAD','cache_before','cache_after','peak_allocated_gib')} for n,r in rows.items()},
    scope='Shared FM8 C1; each model owns KV; same C2 clean history, own generated C3 history. No full 4NFE first window.')
(HERE/f'{args.through}_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
