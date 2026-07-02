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
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    Plot model residuals vs time and vs phase (if period provided).
    """
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    residuals = model_outputs.get("residuals")
    if residuals is None:
        residuals = flux - np.median(flux)

    residuals = np.asarray(residuals)

    n_panels = 2 if period is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), squeeze=False)

    axes[0, 0].scatter(time, residuals, s=0.5, color="gray", alpha=0.5, rasterized=True)
    axes[0, 0].axhline(0, color="red", linewidth=0.8)
    axes[0, 0].set_xlabel("Time (BTJD)")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Residuals vs Time", fontsize=10, fontweight="bold")

    if period is not None:
        phase, _, _ = phase_fold(lc, period, epoch)
        raw_time = np.asarray(lc.time.value)
        t0 = epoch if epoch is not None else raw_time[0]
        raw_phase = ((raw_time - t0) / period) % 1.0
        raw_phase[raw_phase >= 0.5] -= 1.0
        sort_idx = np.argsort(raw_phase)

        axes[1, 0].scatter(
            raw_phase[sort_idx] * period, residuals[sort_idx],
            s=0.5, color="gray", alpha=0.5, rasterized=True,
        )
        axes[1, 0].axhline(0, color="red", linewidth=0.8)
        axes[1, 0].set_xlabel("Time since transit (days)")
        axes[1, 0].set_ylabel("Residuals")
        axes[1, 0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Residuals vs Phase", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_gp_acf(
    lc: Any,
    model_outputs: dict[str, Any],
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    Plot autocorrelation of residuals before and after GP detrending.
    Validates that red noise is properly modeled by the GP.
    """
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    transit_model = model_outputs.get("transit_model")
    if transit_model is None:
        transit_model = np.ones_like(flux)

    res_before = flux - transit_model

    res_after = model_outputs.get("residuals")
    if res_after is None:
        flux_model = model_outputs.get("flux_model")
        if flux_model is not None:
            res_after = flux - flux_model
        else:
            res_after = np.zeros_like(flux)

    fig, ax = plt.subplots(figsize=(10, 4))

    def get_acf(x, max_lag=150):
        x_centered = x - np.mean(x)
        n = len(x_centered)
        acf = np.correlate(x_centered, x_centered, mode='full')[n-1:n+max_lag]
        if acf[0] > 0:
            acf = acf / acf[0]
        return acf

    max_lag = min(len(time) // 4, 150)
    lags = np.arange(max_lag)
    acf_before = get_acf(res_before, max_lag=max_lag-1)
    acf_after = get_acf(res_after, max_lag=max_lag-1)

    ax.plot(lags, acf_before, color="#94a3b8", label="Raw Residuals (No GP)", linewidth=1.5)
    ax.plot(lags, acf_after, color="#0f766e", label="MCMC Residuals (After GP)", linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    conf = 1.96 / np.sqrt(len(time))
    ax.axhspan(-conf, conf, color="#0f766e", alpha=0.1, label="95% Confidence Interval")

    ax.set_xlabel("Lag (indices)")
    ax.set_ylabel("Autocorrelation Coeff")
    ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | GP Residuals Autocorrelation Function (ACF)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def plot_transit_stack(
    lc: Any,
    period: float,
    epoch: float,
    duration_hr: float = 3.0,
    tic_id: str = "",
    sectors_str: str = "",
    gp_model: np.ndarray | None = None
) -> plt.Figure:
    """
    Plot individual transit events stacked vertically to visually verify
    that transit detection is consistent and not driven by outliers or single sectors.
    """
    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    if gp_model is not None:
        detrended = flux - gp_model
    else:
        detrended = flux / np.median(flux)

    duration_days = duration_hr / 24.0
    half_width = max(0.2, duration_days * 3.0)

    t_start = np.min(time)
    t_end = np.max(time)

    n_start = int(np.floor((t_start - epoch) / period))
    n_end = int(np.ceil((t_end - epoch) / period))

    epochs = [epoch + n * period for n in range(n_start, n_end + 1)]
    epochs = [ep for ep in epochs if t_start - half_width < ep < t_end + half_width]

    if len(epochs) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No transits found in data range", ha="center", va="center")
        return fig

    max_to_plot = 8
    if len(epochs) > max_to_plot:
        indices = np.linspace(0, len(epochs) - 1, max_to_plot, dtype=int)
        epochs_to_plot = [epochs[i] for i in indices]
    else:
        epochs_to_plot = epochs

    fig, axes = plt.subplots(len(epochs_to_plot), 1, figsize=(8, 1.5 * len(epochs_to_plot)), sharex=True)
    if len(epochs_to_plot) == 1:
        axes = [axes]

    for idx, ep in enumerate(epochs_to_plot):
        ax = axes[idx]
        mask = (time >= ep - half_width) & (time <= ep + half_width)

        t_sub = time[mask] - ep
        f_sub = detrended[mask]

        if len(t_sub) > 0:
            ax.scatter(t_sub, f_sub, s=1.5, color="black", alpha=0.4, rasterized=True)
            ax.axvline(0, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_ylabel(f"Transit {idx+1}")
        ax.set_xlim(-half_width, half_width)

        if len(f_sub) > 0:
            local_std = np.std(f_sub)
            ax.set_ylim(np.min(f_sub) - 0.002, 1.0 + 3.0 * local_std)

    axes[-1].set_xlabel("Time since transit mid-time (days)")
    fig.suptitle(f"TIC {tic_id} | Sectors: {sectors_str} | Stacked Individual Transits", y=0.99, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.2)
    return fig
