"""CPU-only EXP-025 media comparison and audit from already saved RGB arrays."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP23 = ROOT / "submission/experiments/EXP-023_v3_fm_small_data"
sys.path.insert(0, str(EXP23))

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fm_run as old
from common import inputs, raw_source_rgb, sha, write

ROLES = ("train_constant", "train_switch", "dev_constant", "dev_switch")
KINDS = tuple((role, "GT39_generated17") for role in ROLES) + tuple(
    (role, "self56") for role in ("dev_constant", "dev_switch"))


def read_rgb(phase, name):
    path = HERE / phase / f"{name}.rgb.npy"
    value = np.load(path, allow_pickle=False)
    if value.shape != (56, 480, 832, 3) or value.dtype != np.uint8:
        raise RuntimeError(f"invalid saved RGB: {path}")
    return value, path


def main():
    status = json.loads((HERE / "GPU_LEDGER.json").read_text())
    if status["status"] != "COMPLETE_PENDING_JUDGE":
        raise RuntimeError("GPU work must finish before CPU media finalization")
    if status["counts"] != {"dit": 570, "backward": 256, "update": 256,
                             "vae_decode": 12, "vae_encode": 0, "text": 0}:
        raise RuntimeError("GPU accounting mismatch")
    _, _, _, train, dev, roles = inputs()
    gallery = HERE / "comparisons"
    if gallery.exists():
        raise FileExistsError(gallery)
    gallery.mkdir()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    audit = []
    old.HERE = HERE
    for role, kind in KINDS:
        name = f"{role}_{kind}"
        before, before_path = read_rgb("eval_before", name)
        after, after_path = read_rgb("eval_after", name)
        clip = roles[role]["clip_id"]
        source = raw_source_rgb((train if role.startswith("train") else dev)[clip])
        panels = np.empty((56, 528, 2496, 3), dtype=np.uint8)
        panels[:] = 18
        for t in range(56):
            panels[t, 48:, :832] = source[t]
            panels[t, 48:, 832:1664] = before[t]
            panels[t, 48:, 1664:] = after[t]
            canvas = Image.fromarray(panels[t])
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 10), "GT source", font=font, fill="white")
            draw.text((852, 10), "Full50 zero adapter", font=font, fill="white")
            draw.text((1684, 10), "Full50 256 updates", font=font, fill="white")
            draw.text((2350, 10), f"f{t:02d}", font=font, fill="white")
            panels[t] = np.asarray(canvas)
        video_path = gallery / f"{name}_GT_zero_256.mp4"
        old.save_video(panels, video_path)
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            decoded = sum(1 for _ in container.decode(video=0))
            fps = float(stream.average_rate)
        if decoded != 56 or abs(fps-24) > .001:
            raise RuntimeError(f"comparison decode failed: {video_path}")
        indices = (0, 18, 38, 39, 47, 55)
        sheet = Image.new("RGB", (2496, 528*len(indices)))
        for k, t in enumerate(indices):
            sheet.paste(Image.fromarray(panels[t]), (0, 528*k))
        sheet_path = gallery / f"{name}_contact.png"
        sheet.save(sheet_path)
        audit.append({"role": role, "kind": kind, "clip_id": clip,
                      "source_video": (train if role.startswith("train") else dev)[clip]["source_video"],
                      "pre_rgb_sha256": sha(before_path),
                      "post_rgb_sha256": sha(after_path),
                      "video": str(video_path.relative_to(ROOT)),
                      "video_sha256": sha(video_path),
                      "contact": str(sheet_path.relative_to(ROOT)),
                      "frames": decoded, "fps": fps})
        write(gallery / "MEDIA_AUDIT.json", audit)
        print("COMPARED", name, flush=True)
    print("EXP025_CPU_MEDIA_COMPLETE", len(audit), flush=True)


if __name__ == "__main__":
    main()
