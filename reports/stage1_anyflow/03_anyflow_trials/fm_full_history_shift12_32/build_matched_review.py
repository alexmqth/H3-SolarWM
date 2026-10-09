"""Archive the completed matched FM/AnyFlow control without modifying inputs."""
from pathlib import Path
import hashlib
import json
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ANY = ROOT / 'outputs/2026-10-08-10/stage1_shift12_duration64'
ANY4 = ROOT / 'outputs/2026-10-08-11/stage1_shift12_step32_4step'
TEACHER = ROOT / 'outputs/2026-10-02-03'
REPORT = OUT / 'report'
REPORT.mkdir(exist_ok=True)
FONT = ImageFont.load_default(size=17)

def decode(video):
    with av.open(video) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height, stream.average_rate) == (832, 480, 24)
        frames = [(f.time, f.to_image()) for f in container.decode(video=0)]
    assert len(frames) == 39
    assert all(abs(t-i/24) < 1e-6 for i, (t, _) in enumerate(frames))
    return [f for _, f in frames]

fm = json.loads((OUT / 'run.json').read_text())
af = json.loads((ANY / 'run.json').read_text())
af4 = json.loads((ANY4 / 'run.json').read_text())
assert fm['status'] == af4['status'] == 'complete'
af['evaluations'] = [e for e in af['evaluations'] if e['step'] == 32] + af4['evaluations']
for state in (fm, af):
    assert {(e['step'], e['action'], e['steps_per_chunk']) for e in state['evaluations']} == {(32,a,n) for a in 'AD' for n in (4,8)}
    assert all(e['runtime']['status'] == 'complete' for e in state['evaluations'])
manifest = dict(scope='matched full-history train shift12 32-update FM/AnyFlow diagnostic; NOT ACCEPTED',
                inputs=[], videos=[], contact_sheets=[])
for nfe in (4, 8):
    sources = []
    for action in 'AD':
        teacher = TEACHER / f'action_{action}_teacher_39'
        tr = json.loads((teacher / 'baseline.json').read_text())
        sources.append((action, 'Original H3 30 full steps', teacher / 'baseline.mp4', tr, 30, 0))
        for label, state in [('FM32 full-history', fm), ('AnyFlow32 full-history', af)]:
            row = next(e for e in state['evaluations'] if e['action'] == action and e['steps_per_chunk'] == nfe)
            assert row['input_fairness']['exactly_equal']
            sources.append((action, f'{label} | {nfe} steps/chunk', Path(row['video']), row['runtime'], nfe*3, 3))
    clips = [decode(item[2]) for item in sources]
    for j, (action, label, path, runtime, calls, commits) in enumerate(sources):
        assert runtime['denoiser_forwards'] == calls and runtime.get('commit_forwards', 0) == commits
        manifest['inputs'].append(dict(action=action, label=label, video=str(path),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if j % 3 != 0:
            canvas = Image.new('RGB', (1664, 1810), 'black')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f'{action} | {label} | ALL 39 FRAMES | diagnostic, not accepted', font=FONT, fill='white')
            for i, frame in enumerate(clips[j]):
                x, y = i%5*332, 34+i//5*220
                draw.text((x+4,y), f'frame {i:02d} / {i/24:.3f}s', font=FONT, fill='white')
                canvas.paste(frame.resize((332,192)), (x,y+24))
            sheet = REPORT / f'{"FM32" if j % 3 == 1 else "AnyFlow32"}_{action}_{nfe}step_all39.jpg'
            canvas.save(sheet, quality=95)
            manifest['contact_sheets'].append(str(sheet))
    for action_index, action in enumerate('AD'):
        canvas = Image.new('RGB', (1664, 864), 'black')
        draw = ImageDraw.Draw(canvas)
        for row_index in range(3):
            j = action_index*3+row_index
            for col, index in enumerate((12,24,30,38)):
                x, y = col*416, row_index*288
                draw.text((x+4,y+4), f'{action} {sources[j][1]}', font=FONT, fill='white')
                draw.text((x+4,y+25), f'frame {index}', font=FONT, fill='white')
                canvas.paste(clips[j][index].resize((416,240)), (x,y+48))
        canvas.save(REPORT / f'{action}_{nfe}step_teacher_FM_AnyFlow.jpg', quality=95)
    video = REPORT / f'matched_FM32_AnyFlow32_{nfe}step_AD_diagnostic.mp4'
    with av.open(video, 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 1872, 880, 'yuv420p'
        stream.options = {'crf':'18', 'preset':'medium', 'threads':'4'}
        for i in range(39):
            canvas = Image.new('RGB', (1872,880), 'black')
            draw = ImageDraw.Draw(canvas)
            draw.text((8,8), 'DIAGNOSTIC: NOT ACCEPTED | matched full-history / shift12 / rank / 32 updates | same image, action, seed and noise', font=FONT, fill='#ffcb66')
            for j, (action, label, path, runtime, calls, commits) in enumerate(sources):
                x,y=j%3*624, 42+j//3*404
                draw.text((x+8,y), f'{action} | {label}', font=FONT, fill='white')
                draw.text((x+8,y+22), f'{runtime["wall_including_shared_setup_seconds"]:.1f}s | {calls} noisy + {commits} commits', font=FONT, fill='white')
                canvas.paste(clips[j][i].resize((624,360)), (x,y+44))
            draw.text((8,856), f'Frame {i:02d}/38 | 39f / 24fps | single-run timings; teacher offload differs; no speedup claim', font=FONT, fill='white')
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    with av.open(video) as container:
        assert sum(1 for _ in container.decode(video=0)) == 39
    manifest['videos'].append(dict(path=str(video), frames=39, fps=24,
                                  sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
(REPORT / 'review_artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest['videos'],indent=2))
