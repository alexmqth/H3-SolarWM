"""Independent CPU audit of EXP010 core/branch outputs; visual verdict separate."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import av
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parents[2]
OUT=ROOT/'H3-World/outputs/EXP-010_v3_fm8_sw158'
FM8=ROOT/'H3-World/outputs/EXP-006_v3_fm8_full'
SW=EXP.parent/'EXP-005_v3_sliding_window'
RUNTIME=ROOT/'H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime'
sys.path[:0]=[str(RUNTIME/'code'),str(RUNTIME/'DiffSynth-Studio-h3-v2'),str(SW),str(EXP.parent/'EXP-006_v3_fm8_full')]
from chunk_plan import PLAN,validate_cache
from run_fm8 import tensor_sha
from causal.local_topology import visible_inputs


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

parser=argparse.ArgumentParser();parser.add_argument('--through',choices=['core','branch'],required=True)
args=parser.parse_args();branch=args.through=='branch'
ledger=json.loads((OUT/'budget.json').read_text());assert 'active_start' not in ledger
expected=(62,56,6,7) if branch else (28,24,4,3)
assert tuple(ledger[k] for k in ('forwards','sampling','commits','vae'))==expected
assert len(ledger['runs'])==(8 if branch else 4)
assert all(r['status']=='complete_pending_visual_review' for r in ledger['runs'])
assert ledger['gpu_seconds']<=.75*3600
mp=EXP/'source_manifest.json';manifest=json.loads(mp.read_text());mh=sha(mp)
for name,item in manifest['sources'].items():assert sha(ROOT/item['path'])==item['sha256'],name
fixture=torch.load(ROOT/manifest['sources']['long47']['path'],map_location='cpu',weights_only=True)
sigmas=json.loads((FM8/'AA/chunk_17_22.json').read_text())['sigmas']
base=[torch.load(FM8/x,map_location='cpu',weights_only=True) for x in ('first12.pt','AA/chunk_12_17.pt','AA/chunk_17_22.pt')]
rows={};cache_sha={};pieces=list(base)
stages=[('shared','commit_c3',None,2),('shared','C4',None,3),('shared','C5',None,4),('shared','C6',None,5)]
if branch:stages += [(b,s,b,i) for b in ('A','D') for s,i in [('C7',6),('C8',7)]]
for folder,stage,action,index in stages:
    directory=OUT/folder;label=f'{folder}/{stage}';start,stop=PLAN.span(index)
    rp=directory/f'{stage}{"_"+action if action else ""}.json';row=json.loads(rp.read_text())
    assert row['status']=='complete_pending_visual_review'
    assert row['source_manifest_sha256']==mh and row['runner_sha256']==manifest['sources']['runner']['sha256']
    assert row['index']==index and row['interval']==[start,stop]
    counts=(0,1,0) if stage=='commit_c3' else (8,int(stage!='C8'),1)
    assert tuple(row[k] for k in ('sampling_forwards','commit_forwards','vae_calls'))==counts
    assert row['peak_allocated_gib']<=44 and row['sigmas']==sigmas
    expected_before=list(PLAN.ancestors(index))
    assert row['cache_before']['ancestors']==expected_before
    assert row['cache_before']['layer_count']==50 and row['cache_before']['commits']==index*50
    if stage=='commit_c3':parent_sha=manifest['sources']['fm8_AA_cache17']['sha256'];history=torch.cat(base[:2],dim=2)
    else:
        if stage=='C4':parent='shared/commit_c3'
        elif stage in ('C5','C6'):parent=f'shared/C{index}'
        elif stage=='C7':parent='shared/C6'
        else:parent=f'{folder}/C7'
        parent_sha=cache_sha[parent]
        parts=list(base)
        for a,z in [(22,27),(27,32),(32,37)]:
            if a>=start:break
            parts.append(torch.load(OUT/'shared'/f'chunk_{a}_{z}.pt',map_location='cpu',weights_only=True))
        if stage=='C8':parts.append(torch.load(directory/'chunk_37_42.pt',map_location='cpu',weights_only=True))
        history=torch.cat(parts,dim=2)
        assert history.shape[2]==start
    assert row['source_cache_sha256']==parent_sha
    assert tensor_sha(history.float())==row['history_tensor_sha256']
    layout,prompt=visible_inputs(fixture['packed'],fixture['prompts'][action or 'A'],stop,390)
    assert tensor_sha(prompt)==row['prompt_sha256']
    assert len(layout['action_text_rows'])==stop and layout['seq_len']==int(layout['action_video_start'])+stop*390
    if stage!='C8':
        cp=directory/f'cache_through{stop}.pt';cache_sha[label]=sha(cp);assert cache_sha[label]==row['cache_after_sha256']
        assert row['cache_after']['ancestors']==list(PLAN.ancestors(index+1))
        assert row['cache_after']['layer_count']==50 and row['cache_after']['commits']==(index+1)*50
        if stage in ('C6','C7'):
            # Read the real retained tensors and inspect every layer, not only JSON counts.
            cache=torch.load(cp,map_location='cpu',weights_only=False)
            actual=validate_cache(cache,plan=PLAN,index=index+1,frame_rows=390,expected_layers=50)
            assert actual==row['cache_after'] and actual['nbytes']==14164800000
            del cache
    if stage!='commit_c3':
        assert tensor_sha(fixture['initial_noise'][:,:,start:stop].float())==row['initial_noise_sha256']
        assert row['cache_read_unchanged'] and row['source_cache_unchanged'] and row['old_RGB_unchanged']
        endpoint=directory/f'chunk_{start}_{stop}.pt';assert sha(endpoint)==row['endpoint_sha256']
        n=PLAN.rgb_stop(index);rgbp=directory/f'published_{n}.npy';assert sha(rgbp)==row['published_file_sha256']
        rgb=np.load(rgbp);assert rgb.shape==(n,480,832,3)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['published_RGB_sha256']
        if stage=='C4':prevp=FM8/'AA/published_73.npy'
        elif stage in ('C5','C6','C7'):prevp=OUT/'shared'/f'published_{n-17}.npy'
        else:prevp=directory/'published_141.npy'
        assert sha(prevp)==row['source_published_sha256']
        previous=np.load(prevp);assert np.array_equal(rgb[:len(previous)],previous)
        assert hashlib.sha256(previous.tobytes()).hexdigest()==row['prior_RGB_sha256']
        video=directory/f'rollout_{n}.mp4';assert sha(video)==row['video_sha256']
        with av.open(str(video)) as c:
            assert c.streams.video[0].average_rate==24
            fs=list(c.decode(video=0));pts=[f.pts for f in fs]
            assert len(fs)==n and all(b>a for a,b in zip(pts,pts[1:]))
            assert all((f.width,f.height)==(832,480) for f in fs)
    rows[label]=row
result=dict(task='EXP-010/v1-FM8-SWG',through=args.through,independent_execution_audit='PASS',visual_assessment='SEPARATE',
    forwards=ledger['forwards'],sampling=ledger['sampling'],commits=ledger['commits'],vae=ledger['vae'],updates=0,
    gpu_seconds=ledger['gpu_seconds'],gpu_hours=ledger['gpu_seconds']/3600,
    max_allocated_gib=max(r['peak_allocated_gib'] for r in rows.values()),
    full_cache_tensor_validation='C6 after first eviction'+('; both C7 after second eviction' if branch else ''),
    rows={n:{k:r.get(k) for k in ('sampling_seconds','commit_seconds','decode_seconds','seconds','flow','boundary_gray_MAD','inside_gray_MAD','cache_before','cache_after','peak_allocated_gib')} for n,r in rows.items()})
(HERE/f'{args.through}_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
