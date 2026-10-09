"""Verified real ABot RGB latents; never relabel teacher rollouts as real data."""
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(manifest_path,device,action_adapter=None):
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('format')!='h3world_real_abot_manifest_v1' or manifest.get('source_kind')!='real_ABot_episode':
        raise ValueError('A genuine encoded ABot manifest is required')
    if manifest.get('chunk_frames')!=5 or manifest.get('history_chunks')!=5:
        raise ValueError('Unsupported real-video chunk protocol')
    if manifest.get('anchor_protocol')!='global_retimed_rgb_prefix_last_image_dual_v2':
        raise ValueError('Real-video RGB anchor protocol mismatch')
    cases=[];ids=set()
    for row in manifest['clips']:
        file=Path(row['encoded_file'])
        if row['clip_id'] in ids:raise ValueError('Duplicated clip id')
        ids.add(row['clip_id'])
        if sha(file)!=row['sha256']:raise ValueError(f'Encoded file changed: {file}')
        state=torch.load(file,map_location='cpu',weights_only=True)
        if state.get('format')!='h3world_real_abot_causal_v1' or state.get('source_kind')!='real_ABot_episode':
            raise ValueError('Refusing non-real target latents')
        source=state['source']
        for key in ('clip_id','sample_id','split'):
            if source[key]!=row[key]:raise ValueError(f'Source identity mismatch: {key}')
        for key,hashfield in [('video','video_sha256'),('action','action_sha256'),('first_frame','first_frame_sha256')]:
            if sha(source[key])!=source[hashfield]:raise ValueError('Source RGB/action file changed')
        clean=state['clean_latents'];cond=state['conditioning'];packed=state['causal_packed']
        anchors=state['anchors'];actions=state['action_cond']
        if clean.shape!=(1,24,12,30,52) or actions.shape!=(12,9) or len(anchors)!=3:
            raise ValueError('Invalid real-video shape')
        if len(packed['action_text_spans_local'])!=12 or len(state['action_script'])!=12:
            raise ValueError('Missing per-latent action spans')
        if state['latent_RGB_spans']!=[[0,1],[1,5],[5,9],[9,13],[13,17],[17,18],[18,22],[22,26],[26,30],[30,34],[34,35],[35,39]]:
            raise ValueError('Action/video temporal bins differ')
        tensors=[clean,cond['prompt_embeds'],cond['audio_noise'],actions,*anchors]
        if not all(torch.isfinite(x).all() for x in tensors):raise ValueError('Nonfinite real training inputs')
        def move(x):return x.to(device) if torch.is_tensor(x) else x
        cases.append(dict(label=row['clip_id'],clean=clean.to(device=device,dtype=torch.float32),
            prompt=cond['prompt_embeds'].to(device),packed={k:move(v) for k,v in packed.items()},
            audio=cond['audio_noise'].to(device),anchors=[x.to(device) for x in anchors],
            actions=actions.to(device),action_adapter=action_adapter,chunks=3,split=row['split'],
            sample_id=row['sample_id'],source_kind='real_ABot_episode',source_file=str(file),
            source_sha256=row['sha256'],target_action=source['target']))
    train=[c for c in cases if c['split']=='train'];validation=[c for c in cases if c['split']=='validation']
    if len(train)+len(validation)!=len(cases) or not train or not validation:
        raise ValueError('Expected explicit nonempty train and validation splits')
    if {c['sample_id'] for c in train}&{c['sample_id'] for c in validation}:
        raise ValueError('Episode leakage between train and validation')
    # First two validation cases cover both actions AND both held-out episodes.
    episodes=sorted({c['sample_id'] for c in validation})
    ordered=[]
    for turn in range(4):
        for i,episode in enumerate(episodes):
            target=('A','D')[(turn+i)%2]
            selected={c['label'] for c in ordered}
            options=[c for c in validation if c['sample_id']==episode and c['target_action']==target and c['label'] not in selected]
            if options:ordered.append(options[0])
    # Compare identities, not tensor-valued dictionaries.
    ordered_ids={c['label'] for c in ordered}
    ordered.extend(c for c in validation if c['label'] not in ordered_ids)
    return train,ordered
