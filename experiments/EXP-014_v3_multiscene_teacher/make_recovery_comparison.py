"""CPU-only AA versus recovered AD side-by-side video."""
from __future__ import annotations

import argparse

from common import HERE, OUT
from make_teacher_comparison import make
from recover_ad import CONFIG, RECOVERY


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=CONFIG["scenes"], required=True)
    args = parser.parse_args()
    scene = args.scene
    make(OUT / "G2" / scene / "FM30/AA/rollout_56.mp4",
         RECOVERY / scene / "AD/rollout_56.mp4",
         HERE / "artifacts/recovery_v2_comparisons" / f"{scene}_AA_vs_recovered_AD_56.mp4",
         "V3 FM30 | AA | 30 NFE/chunk", "V3 FM30 | recovered AD | 30 NFE/chunk",
         f"{scene} | same saved C1/cache/noise; C2 action A vs D")
