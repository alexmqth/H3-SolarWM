"""Archive completed coarse-window controls using measured RGB bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import av


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(base,dest,groups):
    summaries=[]
    for group in groups:
        source=base/group;r=json.loads((source/'evaluation.json').read_text())
        assert r['status']=='complete_pending_visual_review'
        proc=Path(f'/proc/{r["pid"]}/stat')
        if proc.exists():
            st=proc.read_text().rsplit(')',1)[1].split()
            assert st[19]!=r['start_ticks'] or st[0] in ('Z','X'),'Process still active'
        assert r['parameter_versions_unchanged'] and r['history_latents_unchanged']
        assert r['optimizer_updates']==0 and r['cpu_KV_MiB']==0
        count=len(r['rgb_bounds'])-1
        assert len(r['records'])==2*count and r['denoiser_forwards']==2*count*30
        for i in range(count):
            a,d=[next(v for v in r['records'] if v['window']==i and v['action']==action) for action in 'AD']
            for field in ['history_sha256','initial_noise_sha256','anchor_sha256','layout_positions_sha256']:
                assert a[field]==d[field],(group,i,field)
            for row in (a,d):
                assert sha(source/Path(row['flow']['path']).name)==row['video_sha256']
        videos=[]
        for path in sorted(source.glob('*.mp4')):
            if path.stem.startswith('window'):
                i=int(path.stem.split('_')[0][6:]);expected=r['rgb_bounds'][i+1]-r['rgb_bounds'][i]
            else:expected=r['rgb_bounds'][-1]
            with av.open(str(path)) as c:
                s=c.streams.video[0];frames=list(c.decode(video=0))
                assert s.codec_context.name=='h264' and s.average_rate==24
                assert len(frames)==expected and all(f.format.name=='yuv420p' for f in frames)
            videos.append(dict(file=path.name,frames=expected,codec='h264',pix_fmt='yuv420p',fps=24,sha256=sha(path)))
        audit=dict(status='complete_full_decode',paired_inputs_verified=True,videos=videos,
                   scope='Technical validity only; manual quality/action review is separate')
        (source/'video_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        target=dest/group;target.mkdir(parents=True,exist_ok=True)
        for p in source.iterdir():
            if p.suffix in ('.json','.mp4','.jpg','.md'):
                shutil.copy2(p,target/p.name);assert sha(p)==sha(target/p.name)
        summaries.append(dict(group=group,windows=count,videos=len(videos),evaluation_sha256=sha(source/'evaluation.json')))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True);p.add_argument('groups',nargs='+');a=p.parse_args()
    archive(a.base,a.dest,a.groups)
