"""CPU display-only relabeling; frozen EXP-004 source comparisons remain intact."""
from pathlib import Path
import hashlib,json
import av
from PIL import ImageDraw,ImageFont
S=Path(__file__).resolve().parents[2]
OUT=S/'report/v2/v2c_strict_causal_kv/8step_continuation/videos'
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
rows=[]
for branch in ['AA','AD']:
 src=S/f'experiments/EXP-004_v2c_8step/artifacts/videos/V3_30_vs_8_{branch}_73.mp4'
 dst=OUT/f'V2c_30_vs_8_{branch}_73.mp4'
 with av.open(str(src)) as inp, av.open(str(dst),'w') as out:
  stream=out.add_stream('libx264',rate=24);stream.width=1664;stream.height=560;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'medium'};n=0
  for frame in inp.decode(video=0):
   im=frame.to_image();draw=ImageDraw.Draw(im);draw.rectangle((0,0,1663,39),fill='#10151d')
   draw.text((10,10),'V2c 30-step/chunk | persistent KV',font=font,fill='white')
   draw.text((842,10),'V2c 8-step new chunks | same weights/KV',font=font,fill='white')
   for packet in stream.encode(av.VideoFrame.from_image(im)):out.mux(packet)
   n+=1
  for packet in stream.encode():out.mux(packet)
 assert n==73
 with av.open(str(dst)) as inp:
  v=inp.streams.video[0];pts=[]
  for i,frame in enumerate(inp.decode(video=0)):
   pts.append(frame.pts)
   if i==56 and branch=='AA':frame.to_image().save('/tmp/V2c_8step_header_check.png')
  assert len(pts)==73 and str(v.average_rate)=='24' and all(a<b for a,b in zip(pts,pts[1:]))
 rows.append(dict(source=str(src.relative_to(S)),target=str(dst.relative_to(S)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),frames=73,fps=24,change='CPU re-encode: top 40-pixel title band V3 -> V2c; full video sequence and history/action footer retained. Original evidence video unchanged.'))
(S/'archive/taxonomy_v2c_20261010/display_video_relabels.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Relabeled and validated 2 display videos; zero GPU inference.')
