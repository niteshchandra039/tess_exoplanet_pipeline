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
    """Plot the raw (stitched but unflattened) TESS light curve, split by sector if gaps exist."""
    
    # Simple gap detection: if there is a gap > 10 days, split into subplots so it looks nice
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    
    gaps = np.diff(time)
    # Find indices where the gap is > 10 days (usually signifies sector breaks)
    gap_indices = np.where(gaps > 10.0)[0]
    
    if len(gap_indices) == 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(time, flux, s=0.5, color="steelblue", alpha=0.6, rasterized=True)
        ax.set_xlabel("Time (BTJD)")
        ax.set_ylabel("PDCSAP Flux")
        ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Raw TESS Light Curve", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig
        
    # Multi-sector plot
    n_panels = len(gap_indices) + 1
    fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(14, 2.5 * n_panels), sharey=True)
    if n_panels == 1:
        axes = [axes]
        
    start_idx = 0
    for i in range(n_panels):
        end_idx = gap_indices[i] if i < len(gap_indices) else len(time) - 1
        
        t_segment = time[start_idx:end_idx+1]
        f_segment = flux[start_idx:end_idx+1]
        
        ax = axes[i]
        ax.scatter(t_segment, f_segment, s=0.5, color="steelblue", alpha=0.6, rasterized=True)
        ax.set_ylabel("PDCSAP Flux")
        
        # Center x-axis per segment to avoid large empty spaces
        if len(t_segment) > 0:
            ax.set_xlim(t_segment[0] - 1, t_segment[-1] + 1)
            
        start_idx = end_idx + 1

    axes[-1].set_xlabel("Time (BTJD)")
    axes[0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Raw TESS Light Curve (Sector-wise)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_clean_lightcurve(lc: Any, tic_id: str = "", sectors_str: str = "") -> plt.Figure:
    """Plot light curve after sigma clipping and NaN removal."""
    return plot_raw_lightcurve(lc, tic_id=tic_id, sectors_str=sectors_str)


def plot_flattened_lightcurve(lc: Any, tic_id: str = "", sectors_str: str = "") -> plt.Figure:
    """Plot the flattened (detrended) light curve, split by sector if gaps exist."""
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    
    gaps = np.diff(time)
    gap_indices = np.where(gaps > 10.0)[0]
    
    if len(gap_indices) == 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(time, flux, s=0.5, color="darkorange", alpha=0.6, rasterized=True)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time (BTJD)")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Flattened TESS Light Curve", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig

    n_panels = len(gap_indices) + 1
    fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(14, 2.5 * n_panels), sharey=True)
    if n_panels == 1:
        axes = [axes]
        
    start_idx = 0
    for i in range(n_panels):
        end_idx = gap_indices[i] if i < len(gap_indices) else len(time) - 1
        
        t_segment = time[start_idx:end_idx+1]
        f_segment = flux[start_idx:end_idx+1]
        
        ax = axes[i]
        ax.scatter(t_segment, f_segment, s=0.5, color="darkorange", alpha=0.6, rasterized=True)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Normalized Flux")
        if len(t_segment) > 0:
             ax.set_xlim(t_segment[0] - 1, t_segment[-1] + 1)
        
        start_idx = end_idx + 1

    axes[-1].set_xlabel("Time (BTJD)")
    axes[0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Flattened TESS Light Curve (Sector-wise)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig
