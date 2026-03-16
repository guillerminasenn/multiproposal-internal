"""Publication-style Matplotlib defaults for reports figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_pub_style() -> None:
    """Apply publication-friendly defaults for figures."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "lines.linewidth": 0.9,
    })
