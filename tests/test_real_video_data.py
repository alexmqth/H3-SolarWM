"""Synthetic I/O fixtures test provenance/split rejection, not video quality."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from causal.real_video_data import load_cases

SPANS=[[0,1],[1,5],[5,9],[9,13],[13,17],[17,18],[18,22],[22,26],[26,30],[30,34],[34,35],[35,39]]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fixture_manifest(tmp_path,*,leak=False,bad_spans=False):
    rows=[]
    for index,(split,episode,target) in enumerate([('train','train_episode','A'),
        ('validation','train_episode' if leak else 'val_episode_1','A'),('validation','val_episode_2','D')]):
        label=f'synthetic_{index}';source=dict(clip_id=label,sample_id=episode,split=split,target=target)
        for field,hashfield in [('video','video_sha256'),('action','action_sha256'),('first_frame','first_frame_sha256')]:
            file=tmp_path/f'{label}_{field}.fixture';file.write_bytes((field+label).encode())
            source[field]=str(file);source[hashfield]=digest(file)
        temporal=[list(x) for x in SPANS]
        if bad_spans and index==0:temporal[5]=[17,21]
        state=dict(format='h3world_real_abot_causal_v1',source_kind='real_ABot_episode',source=source,
            clean_latents=torch.zeros(1,24,12,30,52,dtype=torch.bfloat16),
            conditioning=dict(prompt_embeds=torch.zeros(12,4),audio_noise=torch.zeros(2,32,65)),
            causal_packed=dict(action_text_spans_local=[(i,i+1) for i in range(12)]),
            anchors=[torch.zeros(780,96) for _ in range(3)],action_cond=torch.zeros(12,9),
            action_script=['synthetic fixture']*12,latent_RGB_spans=temporal)
        file=tmp_path/f'{label}.pt';torch.save(state,file)
        rows.append(dict(clip_id=label,sample_id=episode,split=split,encoded_file=str(file),sha256=digest(file)))
    manifest=dict(format='h3world_real_abot_manifest_v1',source_kind='real_ABot_episode',chunk_frames=5,history_chunks=5,
        anchor_protocol='global_retimed_rgb_prefix_last_image_dual_v2',clips=rows)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest));return path,rows

def test_train_validation_separate_and_float32_targets(tmp_path):
    path,_=fixture_manifest(tmp_path);train,val=load_cases(path,'cpu')
    assert len(train)==1 and len(val)==2
    assert {x['sample_id'] for x in train}.isdisjoint({x['sample_id'] for x in val})
    assert [x['target_action'] for x in val]==['A','D']
    assert all(x['clean'].dtype==torch.float32 and x['action_adapter'] is None for x in train+val)

def test_rejects_episode_leakage(tmp_path):
    path,_=fixture_manifest(tmp_path,leak=True)
    with pytest.raises(ValueError,match='Episode leakage'):load_cases(path,'cpu')

def test_rejects_changed_video_after_encoding(tmp_path):
    path,_=fixture_manifest(tmp_path)
    (tmp_path/'synthetic_0_video.fixture').write_bytes(b'changed RGB source')
    with pytest.raises(ValueError,match='Source RGB/action file changed'):load_cases(path,'cpu')

def test_rejects_changed_cache(tmp_path):
    path,rows=fixture_manifest(tmp_path);Path(rows[0]['encoded_file']).write_bytes(b'corrupted cache')
    with pytest.raises(ValueError,match='Encoded file changed'):load_cases(path,'cpu')

def test_rejects_wrong_latent_action_time_bins(tmp_path):
    path,_=fixture_manifest(tmp_path,bad_spans=True)
    with pytest.raises(ValueError,match='temporal bins differ'):load_cases(path,'cpu')

def test_rejects_teacher_generated_dataset(tmp_path):
    p=tmp_path/'manifest.json';p.write_text(json.dumps(dict(format='h3world_real_abot_manifest_v1',source_kind='teacher_rollout')))
    with pytest.raises(ValueError,match='genuine encoded ABot'):load_cases(p,'cpu')
