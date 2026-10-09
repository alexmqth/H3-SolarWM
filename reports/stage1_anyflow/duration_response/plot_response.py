"""Plot the existing signed Farneback measurements; no new quality metric."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = {(x['method'], x['action']): x for x in json.loads(args.input.read_text())['rows']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout='constrained')
    choices = [('Original', 'Original H3, 30 steps', '#222222'),
               ('AF_shift12_32', 'AnyFlow: 32 updates, 8 steps/chunk', '#d55e00'),
               ('AF_shift12_64', 'AnyFlow: 64 updates, 8 steps/chunk', '#0072b2')]
    for ax, action in zip(axes, 'AD'):
        for method, label, color in choices:
            # Include every transition, including boundary transitions.
            values = np.asarray(rows[method, action]['per_transition'])
            assert len(values) == 38
            cumulative = np.r_[0., values.cumsum()]
            ax.plot(np.arange(39), cumulative, label=label, color=color, linewidth=2)
        for boundary in (17, 34):
            ax.axvline(boundary, color='gray', linestyle=':', linewidth=1)
        ax.axhline(0, color='gray', linewidth=.7)
        ax.set(title=f'Action {action}', xlabel='RGB frame index', xlim=(0, 38))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Cumulative signed horizontal flow (px)')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Same image / action / seed; AnyFlow direction gate still fails', fontsize=12)
    fig.supxlabel('Flow computed at 416 x 240. Dotted lines: RGB boundary convention; not isolated latent chunks.', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
