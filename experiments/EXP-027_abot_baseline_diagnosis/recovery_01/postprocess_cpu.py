"""CPU-only media/checksum/paired-review processing of completed EXP-027 videos."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw
import av
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP23 = ROOT / 'submission/experiments/EXP-023_v3_fm_small_data'
sys.path.insert(0, str(EXP23))
from common import inputs, raw_source_rgb


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def media_info(path, width, height):
    with av.open(str(path)) as container:
        streams = container.streams.video
        assert len(streams) == 1
        stream = streams[0]
        assert abs(float(stream.average_rate)-24.0) < .001
        count = 0
        for frame in container.decode(video=0):
            assert (frame.width, frame.height) == (width, height)
            count += 1
    assert count == 56, (path, count)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'frames': count, 'fps': 24, 'width': width, 'height': height,
            'bytes': path.stat().st_size}


def encode(path, frames):
    if path.exists():
        return media_info(path, frames.shape[2], frames.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.mp4')
    assert not temp.exists()
    height, width = frames.shape[1:3]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '24',
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
               '-crf', '18', '-pix_fmt', 'yuv420p', str(temp)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    proc.communicate(np.ascontiguousarray(frames).tobytes())
    if proc.returncode:
        raise RuntimeError(f'ffmpeg failed for {path}')
    media_info(temp, width, height)
    temp.replace(path)
    return media_info(path, width, height)


rows = json.loads((HERE/'VIDEO_RESULTS.json').read_text())
assert len(rows) == 4
_, _, _, train, dev, _ = inputs()
records = []
for clip in sorted({x['clip_id'] for x in rows}):
    row0 = next(x for x in rows if x['clip_id'] == clip and x['checkpoint_updates'] == 0)
    row64 = next(x for x in rows if x['clip_id'] == clip and x['checkpoint_updates'] == 64)
    paths = [ROOT/x['video'] for x in (row0,row64)]
    for row, path in zip((row0,row64), paths):
        assert sha(path) == row['sha256']
        media_info(path, 832, 480)
    before = np.load(paths[0].with_suffix('.rgb.npy'), mmap_mode='r')
    after = np.load(paths[1].with_suffix('.rgb.npy'), mmap_mode='r')
    assert before.shape == after.shape == (56,480,832,3)
    assert np.array_equal(before[:39], after[:39])
    source = raw_source_rgb((train if clip in train else dev)[clip])
    assert np.array_equal(before[:39], source[:39])
    source_raw = HERE/'reference'/f'{clip}_source_rgb56.npy'
    source_raw.parent.mkdir(parents=True,exist_ok=True)
    if source_raw.exists():
        assert np.array_equal(np.load(source_raw,mmap_mode='r'),source)
    else:
        temp_raw = source_raw.with_suffix('.npy.partial')
        assert not temp_raw.exists()
        with temp_raw.open('wb') as stream:
            np.save(stream,source)
        temp_raw.replace(source_raw)
    reference = HERE/'reference'/f'{clip}_source_rgb56.mp4'
    reference_media = encode(reference, source)
    pair = np.empty((56,516,1664,3),dtype=np.uint8)
    pair[:,:36,:,:] = 28
    pair[:,36:,:832] = before
    pair[:,36:,832:] = after
    for frame_index in range(56):
        canvas = Image.fromarray(pair[frame_index])
        draw = ImageDraw.Draw(canvas)
        label = 'GT39 / model C2' if frame_index >=39 else 'GT history'
        draw.text((12,10),f'{clip[:8]}  checkpoint 0  f{frame_index}  {label}',fill='white')
        draw.text((844,10),f'{clip[:8]}  checkpoint 64  f{frame_index}  {label}',fill='white')
        pair[frame_index] = np.asarray(canvas)
    paired_path = HERE/'comparisons'/f'{clip}_checkpoint000_vs_064.mp4'
    paired_media = encode(paired_path, pair)
    def stats(frames):
        gen = np.asarray(frames[39:], dtype=np.float32)
        gt = source[39:].astype(np.float32)
        mae = float(np.abs(gen-gt).mean())
        mse = float(np.square(gen-gt).mean())
        return {'generated_frame_mad':float(np.abs(np.diff(gen,axis=0)).mean()),
                'boundary_mad':float(np.abs(frames[39].astype(np.float32)-frames[38].astype(np.float32)).mean()),
                'source_mae_generated_17f':mae,
                'source_psnr_generated_17f':float(10*np.log10(255**2/mse))}
    records.append({'clip_id':clip,'checkpoint0':stats(before),'checkpoint64':stats(after),
                    'mean_abs_rgb_difference_generated_17f':float(np.abs(before[39:].astype(np.float32)-after[39:].astype(np.float32)).mean()),
                    'reference':reference_media,
                    'reference_lossless_rgb':{'path':str(source_raw.relative_to(ROOT)),
                                              'sha256':sha(source_raw),'shape':list(source.shape)},
                    'paired':paired_media})
    print(clip, records[-1]['checkpoint0'],records[-1]['checkpoint64'],flush=True)

out = HERE/'CPU_REVIEW.json'
out.write_text(json.dumps({'status':'PASS_CPU_MEDIA','records':records},indent=2)+'\n')
print(out)
