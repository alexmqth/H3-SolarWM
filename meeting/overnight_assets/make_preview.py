"""Two stills from complete, already archived FM48 comparisons; no new generation."""
from pathlib import Path
import av
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
base = ROOT/'reports/stage1_anyflow/01_real_video/real_abot_fm/report'
rows = []
for title, suite in [('GT history, 30 steps/chunk — frame 30/38', 'trained_complete_gt30'),
                     ('Generated history, 30 steps/chunk — frame 30/38', 'trained_complete_generated30')]:
    path = base/suite/'dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4'
    with av.open(str(path)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i == 30:
                im = frame.to_image()
                im.thumbnail((1440, 700))
                row = Image.new('RGB', (1440, im.height + 30), '#eeeeee')
                row.paste(im, (0, 30))
                ImageDraw.Draw(row).text((10, 7), title.replace('—', '-'), fill='black')
                rows.append(row)
                break
assert len(rows) == 2
out = Image.new('RGB', (1440, sum(row.height for row in rows)), 'white')
y = 0
for row in rows:
    out.paste(row, (0, y)); y += row.height
out.save(Path(__file__).with_name('real_video_history_gap_frame30.jpg'), quality=92)
