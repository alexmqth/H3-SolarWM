"""Visible-window ABot labels for the native single-I0 E2 experiment.

This format is separate from the old 5-latent/dual-anchor cache. Measured
camera speed F is an observed training proxy, not a recorded user command.
Translation channels are retained for provenance but NEVER condition text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'abot'))
import abot_action as A
import action_script as S

WINDOW_STOPS = (12, 24, 36, 37)
FORMAT = 'h3world_real_transition_windows_v2'


def bounded_keys9(pooled, *, stops=WINDOW_STOPS):
    """Derive each window once using only its own bins; never revise past F.

    Preserve native global unequal bin widths. A narrow bin can borrow its
    next bin WITHIN the current window, otherwise its previous bin within
    that window. A singleton window uses its own observed rate. The caller
    may pass a visible prefix; its end must be a registered window boundary.
    """
    pooled = np.asarray(pooled)
    if pooled.ndim != 2 or pooled.shape[1] != A.ACTION_DIM or not np.isfinite(pooled).all():
        raise ValueError('Expected finite [latent,17] pooled observations')
    if tuple(sorted(set(stops))) != tuple(stops) or not stops or stops[0] <= 0:
        raise ValueError('Strictly increasing positive window boundaries required')
    if len(pooled) not in stops:
        raise ValueError('Visible prefix must end on a declared window boundary')
    widths = np.array([e-s for s, e in A.frame_spans(len(pooled))])
    keys = (pooled[:, A.ACTIVE_KEY_INDICES] > 0).astype(np.float32)
    yaw = pooled[:, A.NUM_KEYS+1]
    rate = yaw / widths
    start = 0
    for stop in stops:
        if stop > len(pooled):
            break
        for k in range(start, stop):
            if widths[k] < 4 and stop-start > 1:
                j = k+1 if k+1 < stop else k-1
                rate[k] = (yaw[k]+yaw[j]) / (widths[k]+widths[j])
        start = stop
    pan = keys[:, S.KEYS9.index('J')] + keys[:, S.KEYS9.index('L')] > 0
    fast = (pan & (np.abs(rate) >= S.YAW_SHARP)).astype(np.float32)
    return np.column_stack((keys, fast))


def window_eligibility(matrix, start, stop, *, min_lateral_fraction=0.5):
    """Conservative eligibility for a lateral action-swap ranking example.

    FM still uses the actual joint action. Negative labels provide NO
    counterfactual video truth. Drop ranking examples with unsupported keys,
    within-bin opposing actions, or fewer than half lateral-active RGB rows.
    """
    matrix = np.asarray(matrix)
    spans = A.frame_spans(37)
    if matrix.shape != (124, 17) or (start, stop) not in ((0, 12), (12, 24), (24, 36)):
        raise ValueError('Expected an observed 124RGB clip and one complete 12-latent window')
    lo, hi = spans[start][0], spans[stop-1][1]
    keys = matrix[lo:hi, :A.NUM_KEYS]
    reasons = []
    unsupported = {k: int(keys[:, A.KEY_COLS.index(k)].sum()) for k in ('Q', 'E', 'Space')}
    if any(unsupported.values()):
        reasons.append('unsupported_key_in_current_window')
    conflicts = []
    for k in range(start, stop):
        left, right = spans[k]
        bits = matrix[left:right, :A.NUM_KEYS].max(axis=0)
        for a, b in (('W','S'), ('A','D'), ('I','K'), ('J','L')):
            if bits[A.KEY_COLS.index(a)] and bits[A.KEY_COLS.index(b)]:
                conflicts.append([k, a, b])
    if conflicts:
        reasons.append('opposing_keys_within_latent_bin')
    lateral = (keys[:, A.KEY_COLS.index('A')] > 0) ^ (keys[:, A.KEY_COLS.index('D')] > 0)
    fraction = float(lateral.mean())
    if fraction < min_lateral_fraction:
        reasons.append('insufficient_lateral_action')
    return dict(latent_start=start, latent_stop=stop, RGB_start=lo, RGB_stop=hi,
                ranking_eligible=not reasons, rejection_reasons=reasons,
                lateral_fraction=fraction, conflicting_bins=conflicts,
                unsupported_counts=unsupported,
                key_counts={key:int(keys[:,i].sum()) for i,key in enumerate(A.KEY_COLS)})


def swap_current_lateral(keys9, start, stop):
    """Only current A and D change; W/S, camera/F and past/future stay exact."""
    keys9 = np.asarray(keys9)
    if keys9.ndim != 2 or keys9.shape[1] != 9 or not 0 <= start < stop <= len(keys9):
        raise ValueError('Invalid key tensor or current window')
    if not np.isin(keys9, (0,1)).all():
        raise ValueError('Expected binary controls')
    a, d = S.KEYS9.index('A'), S.KEYS9.index('D')
    if np.any((keys9[start:stop,a] > 0) & (keys9[start:stop,d] > 0)):
        raise ValueError('Ambiguous opposing lateral keys')
    out = keys9.copy()
    out[start:stop,a] = keys9[start:stop,d]
    out[start:stop,d] = keys9[start:stop,a]
    return out
