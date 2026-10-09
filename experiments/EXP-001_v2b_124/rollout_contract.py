"""Pure EXP-001 schedule and immutable-frame contracts."""

from __future__ import annotations

import hashlib
import numpy as np

PATHS = {"AA": ("A", "A"), "DD": ("D", "D"),
         "AD": ("A", "D"), "DA": ("D", "A")}
RANGES = ((17, 22), (22, 27), (27, 32), (32, 37))
RGB_STOPS = {12: 39, 17: 56, 22: 73, 27: 90, 32: 107, 37: 124}


def rgb_stop(latent_stop: int) -> int:
    return RGB_STOPS[latent_stop]


def prompt_for_path(data, path: str, stop: int):
    """Keep first12 actions, then select the held continuation action.

    The packed layout and original global coordinates come from first action's
    saved fixture. The two fixtures were validated to use the same layout.
    """
    first, later = PATHS[path]
    spans = data[first]["packed"]["action_text_spans_local"]
    text = data[first]["pairs"][0]["prompts"][first].clone()
    donor = data[later]["pairs"][0]["prompts"][later]
    for lo, hi in spans[12:stop]:
        text[lo:hi] = donor[lo:hi]
    return text


def stitch_immutable(published: np.ndarray, full_prefix: np.ndarray, start: int, stop: int):
    """Append only the new 17 RGB frames; never replace old displayed pixels."""
    assert published.dtype == full_prefix.dtype == np.uint8
    assert published.shape[0] == rgb_stop(start)
    assert full_prefix.shape[0] == rgb_stop(stop)
    assert published.shape[1:] == full_prefix.shape[1:]
    return np.concatenate((published, full_prefix[rgb_stop(start):]), axis=0)


def rgb_sha(frames: np.ndarray) -> str:
    assert frames.dtype == np.uint8 and frames.ndim == 4
    return hashlib.sha256(frames.tobytes(order="C")).hexdigest()
