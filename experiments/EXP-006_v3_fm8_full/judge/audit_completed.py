"""Independent completed EXP-006 ledger, output and matching-condition audit."""
from pathlib import Path
import json
import hashlib

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
OUT = ROOT / 'H3-World/outputs/EXP-006_v3_fm8_full'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(4<<20), b''): h.update(x)
    return h.hexdigest()

ledger=json.loads((OUT/'budget.json').read_text())
assert 'active_start' not in ledger
assert (ledger['total_forwards'],ledger['sampling'],ledger['commits'],ledger['vae_decodes']) == (43,40,3,5)
assert ledger['gpu_seconds'] <= 2700
assert len(ledger['runs']) == 5
assert all(r['status']=='complete_pending_visual_review' for r in ledger['runs'])
rows={}
source_manifest=sha(EXP/'source_manifest.json')
for label, folder, start, stop, index in [('C1',OUT,0,12,0),('AA_C2',OUT/'AA',12,17,1),('AA_C3',OUT/'AA',17,22,2),('AD_C2',OUT/'AD',12,17,1),('AD_C3',OUT/'AD',17,22,2)]:
    r=json.loads((folder/f'chunk_{start}_{stop}.json').read_text())
    assert r['status']=='complete_pending_visual_review'
    assert r['sampling_forwards']==8 and r['commit_forwards']==(index<2) and r['vae_calls']==1
    assert r['cache_read_unchanged'] and r['frozen_parameters_unchanged']
    assert r['runner_sha256']==sha(EXP/'run_fm8.py')
    assert r['config_sha256']==sha(EXP/'config.json') and r['manifest_sha256']==source_manifest
    assert r['peak_allocated_gib']<=44
    endpoint=folder/('first12.pt' if index==0 else f'chunk_{start}_{stop}.pt')
    assert sha(endpoint)==r['endpoint_sha256']
    rgb=39+17*index
    assert sha(folder/f'published_{rgb}.npy')==r['published_file_sha256']
    assert sha(folder/f'rollout_{rgb}.mp4')==r['video_sha256']
    check=json.loads((HERE/f'{label}.json').read_text())
    assert check['video_frames']==rgb and check['RGB_sha256']==r['published_RGB_sha256']
    if index:
        assert check['prefix_unchanged'] and r['old_RGB_unchanged'] and r['source_cache_file_unchanged']
        expected=[[0,4680]]+([[1,1950]] if index==2 else [])
        assert r['cache_before']['entries']==expected and r['cache_before']['layer_count']==50
    if index<2:
        assert r['cache_after']['entries']==([[0,4680]] if index==0 else [[0,4680],[1,1950]])
        assert r['cache_after']['layer_count']==50
    rows[label]=r
for index in (2,3):
    for key in ('current_noise_sha256','position_sha256','initial_noise_sha256','audio_sha256','anchor_sha256'):
        assert rows[f'AA_C{index}'][key]==rows[f'AD_C{index}'][key]
assert rows['AA_C2']['cache_file_sha256']==rows['AD_C2']['cache_file_sha256']==rows['C1']['cache_after_sha256']
out=dict(task='EXP-006/v1',independent_audit_passed=True,forwards=43,vae=5,
         gpu_seconds=ledger['gpu_seconds'],gpu_hours=ledger['gpu_seconds']/3600,
         max_allocated_gib=max(r['peak_allocated_gib'] for r in rows.values()),
         rows={k:{q:r.get(q) for q in ('sampling_seconds','commit_seconds','decode_seconds','wall_seconds','peak_allocated_gib','peak_cpu_rss_mib','flow')} for k,r in rows.items()},
         scope='new FM8 first39 plus AA/AD73; C2 same history, C3 own history; requires separate visual judgement')
(HERE/'completed_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
