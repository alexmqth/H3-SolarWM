"""CPU-only C1/C2 visible-layout audit over all EXP-023 encoded conditions."""
from __future__ import annotations

import json
from pathlib import Path

import torch

import coverage as cov

HERE = Path(__file__).resolve().parent


def main():
    selection = json.loads((cov.EXP23 / "SELECTION.json").read_text())
    clips = selection["train_clip_ids"] + [x["clip_id"] for x in selection["evaluation"]
                                             if x["role"].startswith("dev")]
    assert len(clips) == len(set(clips)) == 34
    rows = []
    for clip in clips:
        f = torch.load(cov.EXP23 / "encoded" / clip / "condition.pt",
                       map_location="cpu", weights_only=True)
        assert f["clip_id"] == clip
        data = {"z": torch.empty((1, 24, 17, 30, 52)),
                "full_packed": f["packed"], "prompt": f["prompt"],
                "anchor": f["anchor"], "audio": f["audio_noise"]}
        assert data["anchor"].shape[0] == 390
        out = {"clip_id": clip, "layout": {}}
        for block, stop in (("C1", 12), ("C2", 17)):
            packed, prompt = cov.prepare_visible(data, block)
            prefix = int(packed["action_video_start"])
            assert len(packed["action_text_rows"]) == stop
            assert int(packed["seq_len"]) == prefix + stop * 390
            assert len(prompt) == len(packed["text_pos"])
            assert packed["img_position_ids"].shape[1] == int(packed["seq_len"])
            out["layout"][block] = {"stop_latent": stop,
                                     "visible_action_spans": stop,
                                     "prefix_rows": prefix,
                                     "sequence_rows": int(packed["seq_len"]),
                                     "prompt_rows": len(prompt)}
        rows.append(out)
    result = {"status": "PASS_34_REAL_CPU_VISIBLE_LAYOUTS", "gpu_calls": 0,
              "train_count": 32, "dev_count": 2, "records": rows,
              "note": "Actual EXP-023 encoded condition tensors; latent payload omitted and 33B not loaded."}
    path = HERE / "REAL_INPUT_AUDIT.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], "C1/C2 x", len(rows))


if __name__ == "__main__":
    main()
