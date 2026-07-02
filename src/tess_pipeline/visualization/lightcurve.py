"""
visualization/lightcurve.py — Light curve plotting functions.
"""

from __future__ import annotations

from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["figure.dpi"] = 120


def plot_raw_lightcurve(lc: Any, tic_id: str = "", sectors_str: str = "") -> plt.Figure:
    """Plot the raw (stitched but unflattened) TESS light curve."""
    fig, ax = plt.subplots(figsize=(14, 4))
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    ax.scatter(time, flux, s=0.5, color="steelblue", alpha=0.6, rasterized=True)
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("PDCSAP Flux")
    ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Raw TESS Light Curve", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_clean_lightcurve(lc: Any, tic_id: str = "", sectors_str: str = "") -> plt.Figure:
    """Plot light curve after sigma clipping and NaN removal."""
    return plot_raw_lightcurve(lc, tic_id=tic_id, sectors_str=sectors_str)


def plot_flattened_lightcurve(lc: Any, tic_id: str = "", sectors_str: str = "") -> plt.Figure:
    """Plot the flattened (detrended) light curve."""
    fig, ax = plt.subplots(figsize=(14, 4))
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    ax.scatter(time, flux, s=0.5, color="darkorange", alpha=0.6, rasterized=True)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Flattened TESS Light Curve", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig
