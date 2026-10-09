"""Selectively fetch six real ABot episodes and build aligned 39-frame clips.

No model-generated videos. Episode-disjoint train/validation. Bounded network
payload: six <=256MiB videos and existing ~9MiB annotations, no dataset snapshot.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import av
import numpy as np
from PIL import Image
import requests

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2];DATA=ROOT/'data/abot_bridge'
sys.path.insert(0,str(ROOT/'code/abot'))
import abot_action as A


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def video_url(path):
    prefix='hf://datasets/acvlab/ABot-World-Explorer-500h@'
    assert path.startswith(prefix)
    commit,relative=path[len(prefix):].split('/',1)
    assert len(commit)==40 and relative.startswith('data/')
    return f'https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h/resolve/{commit}/{relative}'


def download(row):
    sid=row['sample_id'];dest=DATA/'raw/data'/sid[:2]/sid/'video.mp4'
    url=video_url(row['video']);budget=256*1024**2
    if not dest.exists():
        with requests.get(url,stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            length=int(response.headers.get('content-length',0))
            if length>budget:raise ValueError(f'Video exceeds bounded download budget: {sid}')
            written=0;temporary=dest.with_suffix('.partial')
            with temporary.open('wb') as f:
                for block in response.iter_content(1024*1024):
                    written+=len(block)
                    if written>budget:raise ValueError('Download budget exceeded')
                    f.write(block)
            if length and written!=length:raise IOError('Incomplete HTTP body')
            temporary.replace(dest)
    with av.open(str(dest)) as container:
        stream=container.streams.video[0]
        assert float(stream.average_rate)==30.,stream.average_rate
        dims=[stream.width,stream.height]
    return dict(sample_id=sid,source_url=url,bytes=dest.stat().st_size,
        sha256=sha(dest),width_height=dims,source_fps=30.)


def candidates(ep):
    offsets=np.asarray(A.window_offsets(39));span=int(offsets[-1])+1
    keys=ep['keys'];all_candidates=[]
    for target in ('A','D'):
        opposite='D' if target=='A' else 'A';ci=A.KEY_COLS.index(target);oi=A.KEY_COLS.index(opposite)
        choices=[]
        for start in range(0,ep['total_frames']-span+1,5):
            rows=keys[start+offsets]
            hit=float(rows[:,ci].mean());wrong=float(rows[:,oi].mean())
            score=hit-wrong
            if hit>=.65 and wrong<=.1:
                choices.append((score,hit,start,target))
        all_candidates.extend(choices)
    # Four nonoverlapping clips/episode, two A and two D when available.
    selected=[]
    for target in ('A','D','A','D'):
        choices=sorted((r for r in all_candidates if r[3]==target),reverse=True)
        for score,hit,start,label in choices:
            if all(abs(start-r['src_start'])>=span for r in selected):
                selected.append(dict(src_start=start,target=label,action_fraction=hit,score=score));break
        else:raise ValueError(f'Insufficient disjoint {target} windows')
    return selected


def prepare_episode(index,row,ffmpeg):
    sid=row['sample_id'];raw=DATA/'raw/data'/sid[:2]/sid
    ep=A.read_episode(str(raw/'annotations.tar'))
    with tarfile.open(raw/'annotations.tar') as tar:
        action=json.load(tar.extractfile('action.json'))
    assert action['fps']==30 and action['sample_stride']==1
    assert len(action['frames'])==1800 and ep['total_frames']==1800
    timestamps=np.array([r['timestamp'] for r in action['frames']])
    assert np.all(np.diff(timestamps)>0)
    assert np.max(np.abs(np.diff(timestamps)-1/30))<1e-4
    assert ep['control_scheme']=='WASD_QE_locomotion_IJKL_rotation'
    split='train' if index<4 else 'validation'
    out=DATA/'clips'/split/sid;out.mkdir(parents=True,exist_ok=True)
    planned=candidates(ep);records=[]
    offsets=np.asarray(A.window_offsets(39));span=int(offsets[-1])+1
    scale=A.episode_translation_scale(ep)
    for item in planned:
        start=item['src_start'];name=f"{item['target']}_{start:04d}"
        video=out/f'{name}.mp4';npy=out/f'{name}.npy';frame=out/f'{name}.png'
        if not video.exists():
            filt=(f"select='between(n\\,{start}\\,{start+span-1})*not(eq(mod(n-{start}\\,5)\\,4))',"
                  'setpts=N/24/TB,scale=-2:480,crop=832:480')
            temporary=video.with_suffix('.tmp.mp4')
            subprocess.run([ffmpeg,'-y','-v','error','-threads','2','-i',str(raw/'video.mp4'),
                '-vf',filt,'-r','24','-frames:v','39','-c:v','libx264','-crf','14',
                '-preset','veryfast','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(temporary)],
                check=True,capture_output=True)
            temporary.replace(video)
        matrix=A.window_action_matrix(ep,start,39,scale)
        np.save(npy,matrix)
        assert np.array_equal(matrix[:,:A.NUM_KEYS],ep['keys'][start+offsets])
        with av.open(str(video)) as c:
            stream=c.streams.video[0];assert float(stream.average_rate)==24.
            frames=[x.to_ndarray(format='rgb24') for x in c.decode(video=0)]
        assert len(frames)==39 and frames[0].shape==(480,832,3)
        Image.fromarray(frames[0]).save(frame)
        counts={key:int(matrix[:,A.KEY_COLS.index(key)].sum()) for key in ('W','A','S','D')}
        prompt=ep['caption']['scene_static'].strip();assert len(prompt.split())>=30
        records.append(dict(sample_id=sid,split=split,clip_id=f'{sid}_{name}',**item,
            video=str(video),action=str(npy),first_frame=str(frame),prompt=prompt,
            source_video_sha256=sha(raw/'video.mp4'),source_annotations_sha256=sha(raw/'annotations.tar'),
            source_frame_indices=(start+offsets).tolist(),RGB_frames=39,latent_frames=12,
            RGB_fps=24.,source_fps=30.,action_counts=counts,
            pooled_action_shape=list(A.bin_to_latent(matrix,12).shape),
            video_sha256=sha(video),action_sha256=sha(npy),first_frame_sha256=sha(frame),
            source_kind='real_ABot_episode',frame_action_same_source_indices=True,
            duplicated_adjacent_RGB_frames=sum(np.array_equal(x,y) for x,y in zip(frames,frames[1:]))))
    return records


def main():
    rows=json.loads((DATA/'annotation_pilot.json').read_text())
    assert len(rows)==6 and len({r['sample_id'] for r in rows})==6
    state=dict(status='downloading_selected_videos',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
        downloads=[],clips=[],byte_budget=6*256*1024**2,episode_count=6)
    def save():
        temp=DATA/'preparation.tmp.json';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(DATA/'preparation.json')
    save()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for d in pool.map(download,rows):state['downloads'].append(d);save();print(json.dumps(d),flush=True)
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        state['status']='building_aligned_clips';save()
        for index,row in enumerate(rows):
            records=prepare_episode(index,row,ffmpeg);state['clips'].extend(records);save()
            print(json.dumps(dict(sample_id=row['sample_id'],clips=len(records))),flush=True)
        train={r['sample_id'] for r in state['clips'] if r['split']=='train'}
        val={r['sample_id'] for r in state['clips'] if r['split']=='validation'}
        assert train.isdisjoint(val)
        (DATA/'clips.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in state['clips']))
        state.update(status='complete',train_clips=sum(r['split']=='train' for r in state['clips']),
            validation_clips=sum(r['split']=='validation' for r in state['clips']),episode_split_disjoint=True,
            total_downloaded_video_bytes=sum(r['bytes'] for r in state['downloads']),
            note='Prepared RGB/action pairs only; latent encoding, causal FM training and evaluation still required')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
