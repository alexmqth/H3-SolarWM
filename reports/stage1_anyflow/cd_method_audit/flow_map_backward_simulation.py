"""Isolated CPU-verified FMBS primitive; not wired into H3 training.

Method: AnyFlow, arXiv:2605.13724, section4.2.2 / Algorithm2.
Use a finite-map velocity callback in the noise-minus-clean convention.
The three nonzero transitions retain their entire autograd chain. Historical
chunks/KV, teacher and critic ownership belong to the future H3 integration.
"""
import math


def shortcut_intervals(sigmas, index):
    times = tuple(float(x) for x in sigmas)
    if (len(times) < 2 or times[0] != 1 or times[-1] != 0
            or any(not math.isfinite(x) for x in times)
            or any(not 0 <= b < a <= 1 for a, b in zip(times, times[1:]))):
        raise ValueError('Require a strictly decreasing 1-to-0 sigma grid')
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(times) - 1:
        raise ValueError('Gradient interval index lies outside the grid')
    candidates = ((times[0], times[index]), (times[index], times[index + 1]),
                  (times[index + 1], times[-1]))
    return tuple((t, r) for t, r in candidates if t != r)


def simulate_chunk(noise, velocity, sigmas, index):
    """T→t→r→0 with at most3 network calls, without detaching a segment.

    For a shifted inference grid, t/r are actual adjacent shifted endpoints.
    This is a shortcut approximation; it is not asserted equal to N steps of
    an imperfect model. The same immutable conditions must wrap every call.
    """
    current = noise.float()
    for t, r in shortcut_intervals(sigmas, index):
        prediction = velocity(current, t, r)
        if prediction.shape != current.shape:
            raise ValueError('Finite-map velocity must match current latent shape')
        current = current + (r - t) * prediction.float()
    return current
