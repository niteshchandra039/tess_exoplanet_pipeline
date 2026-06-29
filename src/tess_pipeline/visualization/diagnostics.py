"""
visualization/diagnostics.py — Residual and convergence diagnostic plots.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tess_pipeline.transit.phasefold import phase_fold


def plot_residuals(
    lc: Any,
    model_outputs: dict[str, Any],
    period: float | None = None,
    epoch: float | None = None,
) -> plt.Figure:
    """
    Plot model residuals vs time and vs phase (if period provided).
    """
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    residuals = model_outputs.get("residuals")
    if residuals is None:
        # Compute from raw flux if no model available
        residuals = flux - np.median(flux)

    residuals = np.asarray(residuals)

    n_panels = 2 if period is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), squeeze=False)

    axes[0, 0].scatter(time, residuals, s=0.5, color="gray", alpha=0.5, rasterized=True)
    axes[0, 0].axhline(0, color="red", linewidth=0.8)
    axes[0, 0].set_xlabel("Time (BTJD)")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Time")

    if period is not None:
        phase, _, _ = phase_fold(lc, period, epoch)
        # Use pre-computed residuals but need to re-align with phase sort
        # Get sort order consistent with phase_fold
        raw_time = np.asarray(lc.time.value)
        t0 = epoch if epoch is not None else raw_time[0]
        raw_phase = ((raw_time - t0) / period) % 1.0
        raw_phase[raw_phase >= 0.5] -= 1.0
        sort_idx = np.argsort(raw_phase)

        axes[1, 0].scatter(
            raw_phase[sort_idx], residuals[sort_idx],
            s=0.5, color="gray", alpha=0.5, rasterized=True,
        )
        axes[1, 0].axhline(0, color="red", linewidth=0.8)
        axes[1, 0].set_xlabel("Phase")
        axes[1, 0].set_ylabel("Residuals")
        axes[1, 0].set_title("Residuals vs Phase")

    fig.tight_layout()
    return fig
