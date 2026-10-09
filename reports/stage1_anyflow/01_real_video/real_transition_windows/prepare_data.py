"""Rebuild observed124RGB transitions, no downloads or model-generated labels."""
from collections import defaultdict
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'code'))
from causal.real_transition_data import (A,S,FORMAT,bounded_keys9,window_eligibility,swap_current_lateral)
DATA=ROOT/'data/abot_bridge'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def dump(path,data):
    tmp=path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(path)


def main():
    dest=BASE/'data'; dest.mkdir(exist_ok=False)
    prep=json.loads((DATA/'preparation.json').read_text())
    started=time.perf_counter()
    state=dict(format=FORMAT,status='auditing',at=datetime.now().astimezone().isoformat(),
               pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
               source_sha256=sha(__file__),input_manifest_sha256=sha(DATA/'preparation.json'),
               source_kind='real_ABot_episode',new_download_bytes=0,GPU_calls=0,optimizer_updates=0,
               clips=[],candidates=[],selection_policy='At most2 per episode, nonoverlapping full124 source spans; maximize eligible later windows then all windows, tie earlier starts. At most8train+4validation. No model results.',
               preprocessing='Decode original source frame to RGB; PIL bicubic scale to height480, nearest-even width854, center crop832; lossless npz for training, JPEG review only.',
               action_policy='Native global bins1/4/4/4/4; camera F neighbor bounded within12latent window; commit past rows once. Raw unnormalized pose translation stored, never used as text conditioning.',
               camera_speed_caveat='F is measured current-window consequence proxy, not an independently recorded exogenous speed command; preserve in A/D negative and report separately.',
               counterfactual_caveat='Swapped action has no observed video truth; ranking is statistical transition discrimination, not paired causal-effect GT.')
    def save():
        state['wall_seconds']=time.perf_counter()-started; dump(BASE/'preparation.json',state)
    save()
    try:
        groups=defaultdict(list); episodes={}; annotations={}
        for row in prep['clips']:
            sid=row['sample_id']; raw=DATA/'raw/data'/sid[:2]/sid
            if sid not in episodes:
                assert sha(raw/'video.mp4')==row['source_video_sha256']
                assert sha(raw/'annotations.tar')==row['source_annotations_sha256']
                ep=A.read_episode(str(raw/'annotations.tar')); episodes[sid]=ep
                import tarfile
                with tarfile.open(raw/'annotations.tar') as t: act=json.load(t.extractfile('action.json'))
                assert act['fps']==30 and act['sample_stride']==1 and len(act['frames'])==ep['total_frames']
                timestamps=np.array([f['timestamp'] for f in act['frames']])
                assert np.max(np.abs(np.diff(timestamps)-1/30))<1e-4
                assert len(set(f['frame_id'] for f in act['frames']))==ep['total_frames']
                annotations[sid]=act
            ep=episodes[sid]; start=row['src_start']
            if start+A.window_offsets(124)[-1]>=ep['total_frames']: continue
            # Scale1 retains raw arbitrary COLMAP translation for audit only.
            matrix=A.window_action_matrix(ep,start,124,1.)
            windows=[window_eligibility(matrix,s,e) for s,e in ((0,12),(12,24),(24,36))]
            candidate=dict(clip_id=row['clip_id'],sample_id=sid,split=row['split'],src_start=start,windows=windows)
            state['candidates'].append(candidate)
            groups[sid].append((candidate,row,matrix))
        selected=[]
        for sid,choices in groups.items():
            sets=[c for n in (1,2) for c in itertools.combinations(choices,n)
                  if all(abs(a[0]['src_start']-b[0]['src_start'])>=A.window_span(124) for a,b in itertools.combinations(c,2))]
            def score(items):
                windows=[w for item in items for w in item[0]['windows']]
                return (sum(w['ranking_eligible'] for w in windows if w['latent_start']>0),
                        sum(w['ranking_eligible'] for w in windows),len(items),
                        tuple(-x[0]['src_start'] for x in sorted(items,key=lambda x:x[0]['src_start'])))
            selected.extend(max(sets,key=score))
        selected.sort(key=lambda x:(x[0]['split'],x[0]['sample_id'],x[0]['src_start']))
        assert sum(c[0]['split']=='train' for c in selected)<=8
        assert sum(c[0]['split']=='validation' for c in selected)<=4
        state['selected_ids']=[c[0]['clip_id'] for c in selected];state['status']='extracting';save()
        for candidate,row,matrix in selected:
            sid=row['sample_id'];raw=DATA/'raw/data'/sid[:2]/sid;start=row['src_start']
            ids=np.array([start+o for o in A.window_offsets(124)])
            frames=[];actual=[];pts=[];wanted=set(ids.tolist())
            with av.open(str(raw/'video.mp4')) as c:
                stream=c.streams.video[0];assert float(stream.average_rate)==30
                for index,f in enumerate(c.decode(video=0)):
                    if index in wanted:
                        im=f.to_image().convert('RGB');sw=round(im.width*480/im.height/2)*2
                        assert sw>=832
                        im=im.resize((sw,480),Image.Resampling.BICUBIC)
                        im=im.crop(((sw-832)//2,0,(sw-832)//2+832,480))
                        frames.append(np.array(im));actual.append(index);pts.append(float(f.pts*stream.time_base))
                    if index>=ids[-1]:break
            assert actual==ids.tolist() and len(frames)==124
            np.testing.assert_allclose(pts,np.array(actual)/30,atol=1e-5,rtol=0)
            np.testing.assert_array_equal(matrix[:,:11],episodes[sid]['keys'][ids])
            pooled=A.bin_to_latent(matrix,37);k9=bounded_keys9(pooled);script=S.annotate_from_keys9(k9)
            # Audit actual examples, including later camera rows and translation.
            for stop in (12,24,36):
                cutoff=A.frame_spans(37)[stop-1][1]
                alt=matrix.copy();alt[cutoff:,:11]=1-alt[cutoff:,:11];alt[cutoff:,11:]*=-100
                np.testing.assert_array_equal(bounded_keys9(A.bin_to_latent(alt,37))[:stop],k9[:stop])
                np.testing.assert_array_equal(bounded_keys9(A.bin_to_latent(matrix[:cutoff],stop)),k9[:stop])
            f=dest/(row['clip_id']+'.npz')
            np.savez_compressed(f,rgb=np.stack(frames),raw_actions=matrix,pooled_actions=pooled,keys9=k9,
                                source_indices=ids,source_pts=np.array(pts),R_cw=episodes[sid]['R_cw'][ids],
                                C_world=episodes[sid]['C_world'][ids])
            sheet=Image.new('RGB',(3*416,3*264),'white');draw=ImageDraw.Draw(sheet)
            for i,j in enumerate((0,38,39,58,80,81,99,119,123)):
                x=(i%3)*416;y=(i//3)*264
                sheet.paste(Image.fromarray(frames[j]).resize((416,240)),(x,y+24));draw.text((x+5,y+5),f'GT RGB {j} / source {ids[j]}',fill='black')
            sheet_path=dest/(row['clip_id']+'_review.jpg');sheet.save(sheet_path,quality=92)
            negatives={}
            for w in candidate['windows']:
                if w['ranking_eligible']:
                    n=swap_current_lateral(k9,w['latent_start'],w['latent_stop'])
                    negatives[str(w['latent_start'])]=S.annotate_from_keys9(n)
            record=dict(**candidate,path=str(f),sha256=sha(f),bytes=f.stat().st_size,
                        review=str(sheet_path),review_sha256=sha(sheet_path),prompt=row['prompt'],
                        source_video_sha256=row['source_video_sha256'],source_annotations_sha256=row['source_annotations_sha256'],
                        source_frame_ids=[annotations[sid]['frames'][i]['frame_id'] for i in ids],
                        source_indices=actual,latent_RGB_spans=A.frame_spans(37),
                        action_script=script,negative_scripts=negatives,
                        original_vs_bounded_F_differences=np.flatnonzero(S.keys9(pooled)[:,-1]!=k9[:,-1]).tolist(),
                        frame_action_same_source_indices=True,temporal_effect_lag='Not independently inferred; retain dataset alignment, no shifted-label search',
                        duplicated_adjacent_RGB=sum(np.array_equal(a,b) for a,b in zip(frames,frames[1:])))
            state['clips'].append(record);save()
            print(json.dumps(dict(clip=row['clip_id'],eligible=[w['latent_start'] for w in candidate['windows'] if w['ranking_eligible']],MB=f.stat().st_size/1e6)),flush=True)
        splits={s:{r['sample_id'] for r in state['clips'] if r['split']==s} for s in ('train','validation')}
        assert splits['train'].isdisjoint(splits['validation'])
        state.update(status='complete_pending_GT_review',episode_split_disjoint=True,full124_source_spans_nonoverlapping_within_episode=True,
                     counts={s:sum(r['split']==s for r in state['clips']) for s in splits},
                     total_bytes=sum(r['bytes'] for r in state['clips']),
                     remaining='GT visual review, VAE/text encoding and future-prefix checks, differentiable forward and memory preflight. No optimization.')
        save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
