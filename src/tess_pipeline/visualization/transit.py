"""
visualization/transit.py — Phase-fold and transit model overlay plots.
"""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from tess_pipeline.transit.phasefold import bin_phase_curve, phase_fold


def plot_phase_curve(
    lc: Any,
    period: float,
    epoch: float | None = None,
    model: dict[str, Any] | None = None,
    n_bins: int = 200,
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    Plot a phase-folded light curve with an optional analytic model overlay.
    """
    phase, flux, flux_err = phase_fold(lc, period, epoch)
    bin_phase, bin_flux, bin_err = bin_phase_curve(phase, flux, flux_err, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw scatter
    ax.scatter(phase * period, flux, s=1.0, color="gray", alpha=0.2, rasterized=True, label="Data")

    # Binned
    valid = np.isfinite(bin_flux)
    ax.errorbar(
        bin_phase[valid] * period, bin_flux[valid], yerr=bin_err[valid],
        fmt="o", ms=4, color="#16a34a", elinewidth=0.8, capsize=2, label="Binned",
    )

    # Model overlay
    depth = 0.005
    if model is not None and model.get("time") is not None:
        model_flux = model.get("transit_model") if model.get("transit_model") is not None else model.get("flux_model")
        if model_flux is not None:
            import lightkurve as lk
            m_lc = lk.LightCurve(time=model["time"], flux=model_flux)
            _, t_fold, _ = phase_fold(m_lc, period, epoch)
            sort_idx = np.argsort(phase)
            x_sorted = phase[sort_idx] * period
            y_sorted = t_fold[sort_idx]
            
            # Interpolate onto a dense uniform grid to avoid gap-induced line segment artifacts
            grid_x = np.linspace(-0.3, 0.3, 1000)
            grid_y = np.interp(grid_x, x_sorted, y_sorted, left=1.0, right=1.0)
            
            ax.plot(
                grid_x, grid_y,
                color="#dc2626", linewidth=2.0, zorder=5, label="Model",
            )
            depth = max(1.0 - np.min(grid_y), 1e-4)
    else:
        # Estimate depth from the binned data around the center phase ([-0.1, 0.1] days)
        near_center = (bin_phase * period >= -0.1) & (bin_phase * period <= 0.1)
        if np.any(near_center) and np.any(valid):
            bin_min = np.nanmin(bin_flux[near_center & valid])
            depth = max(1.0 - bin_min, 1e-4)
            if depth > 0.05 or depth < 1e-5:
                depth = 0.005

    # Zoom y limits to focus on transit
    ax.set_ylim(1.0 - 2.5 * depth, 1.0 + 1.5 * depth)
    ax.set_xlim(-0.3, 0.3)
    ax.set_xlabel("Time since transit (days)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Phase-Folded Transit (P = {period:.5f} d)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def plot_batman_overlay(
    lc: Any,
    period: float,
    epoch: float | None,
    batman_result: dict[str, Any],
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """Plot batman analytic model overlay on phase-folded data."""
    return plot_phase_curve(lc, period, epoch, model=batman_result, tic_id=tic_id, sectors_str=sectors_str)


def plot_bayesian_fit(
    lc: Any,
    posterior: Any,
    model_outputs: dict[str, Any] | None,
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    Plot the Bayesian posterior median transit fit.

    Generates a 2-panel time-series (BJD on x-axis) diagnostic plot:
      1. Normalized flux with GP systematics model trend overlaid.
      2. De-trended flux with Keplerian MCMC transit model overlaid.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)

    time_raw = np.asarray(lc.time.value, dtype=float)
    flux_raw = np.asarray(lc.flux.value, dtype=float)
    sidx = np.argsort(time_raw)
    time_sorted = time_raw[sidx]
    flux_sorted = flux_raw[sidx]
    uniq_mask = np.concatenate([[True], np.diff(time_sorted) > 0])
    time = time_sorted[uniq_mask]
    flux = flux_sorted[uniq_mask]

    gp_model = model_outputs.get("gp_model") if model_outputs else None
    transit_model = model_outputs.get("transit_model") if model_outputs else None
    flux_model = model_outputs.get("flux_model") if model_outputs else None

    # ── Panel 1: Data & GP Systematics (BJD/BTJD on x-axis) ──
    axes[0].scatter(time, flux, s=0.4, color="black", alpha=0.25, label="Data", rasterized=True)
    if gp_model is not None:
        if transit_model is not None:
            trend = flux_model - (transit_model - 1.0)
        else:
            trend = gp_model + np.median(flux)
        axes[0].plot(time, trend, color="#22c55e", linewidth=1.2, label="GP Activity Model")
    axes[0].set_ylabel("Relative Flux")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Bayesian GP Detrending (Time Domain)", fontsize=10, fontweight="bold")

    # ── Panel 2: De-trended Light Curve & MCMC Transit Fit (BJD/BTJD on x-axis) ──
    if gp_model is not None:
        detrended = flux - gp_model
    else:
        detrended = flux
        
    axes[1].scatter(time, detrended, s=0.4, color="black", alpha=0.25, label="Detrended Data", rasterized=True)
    if transit_model is not None:
        axes[1].plot(time, transit_model, color="#dc2626", linewidth=1.2, label="Keplerian Fit")
        
    axes[1].set_ylabel("Relative Flux")
    axes[1].set_xlabel("Time (BTJD)")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_title("Detrended Light Curve & Best-Fit Transit Model", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_mcmc_phase_curve(
    lc: Any,
    posterior: Any,
    period: float,
    epoch: float | None = None,
    n_bins: int = 80,
    planet_idx: int = 0,
    tic_id: str = "",
    sectors_str: str = ""
) -> plt.Figure:
    """
    Plot the phase-folded light curve and residuals using MCMC posterior samples.
    
    Generates a 2-panel plot:
      1. Phase-folded de-trended data with the MCMC transit model overlaid.
      2. Phase-folded residual scatter directly underneath.
    """
    import lightkurve as lk

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    time_raw = np.asarray(lc.time.value, dtype=float)
    flux_raw = np.asarray(lc.flux.value, dtype=float)
    flux_err_raw = np.asarray(lc.flux_err.value, dtype=float) if lc.flux_err is not None else np.ones_like(flux_raw) * 1e-3
    sidx = np.argsort(time_raw)
    time_sorted = time_raw[sidx]
    flux_sorted = flux_raw[sidx]
    flux_err_sorted = flux_err_raw[sidx]
    uniq_mask = np.concatenate([[True], np.diff(time_sorted) > 0])
    time = time_sorted[uniq_mask]
    flux = flux_sorted[uniq_mask]
    flux_err = flux_err_sorted[uniq_mask]

    # Stack the posterior to get flat samples
    post_group = posterior.posterior
    if hasattr(post_group, "to_dataset"):
        ds = post_group.to_dataset()
    else:
        ds = post_group
    flat_samps = ds.stack(sample=("chain", "draw"))

    # Extract GP model prediction (if it exists)
    if "gp_pred" in flat_samps.data_vars:
        gp_mod = np.median(flat_samps["gp_pred"].values, axis=-1)
    else:
        gp_mod = 0.0

    # Extract and subtract other planets' transit models (pre-whitening)
    other_transit_mod = np.zeros_like(flux)
    j = 0
    while True:
        key = f"light_curve_p{j}"
        if key in flat_samps.data_vars:
            if j != planet_idx:
                other_transit_mod += np.median(flat_samps[key].values, axis=-1)
            j += 1
        else:
            break

    # De-trend the data
    detrended_flux = flux - gp_mod - other_transit_mod

    # Extract MCMC derived period and epoch for this planet
    if "period" in flat_samps.data_vars:
        if flat_samps["period"].values.ndim == 2:
            period_med = np.median(flat_samps["period"].values[planet_idx, :])
            period_std = np.std(flat_samps["period"].values[planet_idx, :])
            epoch_med = np.median(flat_samps["t0"].values[planet_idx, :])
        else:
            period_med = np.median(flat_samps["period"].values)
            period_std = np.std(flat_samps["period"].values)
            epoch_med = np.median(flat_samps["t0"].values)
    else:
        period_med = period
        period_std = 0.0
        epoch_med = epoch or time[0]

    # Phase-fold the de-trended data
    detrended_lc = lk.LightCurve(time=time, flux=detrended_flux, flux_err=flux_err)
    phase, y_fold, y_fold_err = phase_fold(
        detrended_lc,
        period=period_med,
        epoch=epoch_med,
    )

    # Plot raw scatter in Top Panel
    axes[0].scatter(phase * period_med, y_fold, s=0.8, color="black", alpha=0.2, label="Data", rasterized=True, zorder=-1000)

    # Plot binned data in Top Panel
    bin_phase, bin_flux, bin_err = bin_phase_curve(phase, y_fold, y_fold_err, n_bins=n_bins)
    valid = np.isfinite(bin_flux)
    axes[0].errorbar(
        bin_phase[valid] * period_med, bin_flux[valid], yerr=bin_err[valid],
        fmt="o", ms=4, color="#16a34a", elinewidth=0.8, capsize=2, label="Binned", zorder=100
    )

    # Get transit model fit
    transit_med = None
    if f"light_curve_p{planet_idx}" in flat_samps.data_vars:
        transit_med = np.median(flat_samps[f"light_curve_p{planet_idx}"].values, axis=-1)
        
        # Check if the dense phase prediction is available to plot a smooth, gap-free model
        lc_pred_key = f"lc_pred_p{planet_idx}"
        if lc_pred_key in flat_samps.data_vars:
            pred_vals = np.median(flat_samps[lc_pred_key].values, axis=-1)
            # The phase grid used during Bayesian model prediction is -0.3 to 0.3 days
            phase_grid = np.linspace(-0.3, 0.3, len(pred_vals))
            axes[0].plot(phase_grid, pred_vals, color="#dc2626", linewidth=2.0, label="Transit Model", zorder=1000)
            depth = max(1.0 - np.min(pred_vals), 1e-4)
        else:
            # Fallback to folding and interpolating the discrete time-series model to avoid gaps
            transit_lc = lk.LightCurve(time=lc.time, flux=transit_med + 1.0, flux_err=lc.flux_err)
            _, t_fold, _ = phase_fold(
                transit_lc,
                period=period_med,
                epoch=epoch_med,
            )
            sort_idx = np.argsort(phase)
            x_sorted = phase[sort_idx] * period_med
            y_sorted = t_fold[sort_idx]
            
            grid_x = np.linspace(-0.3, 0.3, 1000)
            grid_y = np.interp(grid_x, x_sorted, y_sorted, left=1.0, right=1.0)
            
            axes[0].plot(grid_x, grid_y, color="#dc2626", linewidth=2.0, label="Transit Model", zorder=1000)
            depth = max(1.0 - np.min(grid_y), 1e-4)
    else:
        depth = max(1.0 - np.percentile(y_fold, 1), 1e-4)

    # Annotate the period
    txt = f"P = {period_med:.5f} ± {period_std:.5f} d"
    axes[0].annotate(
        txt, (0.02, 0.05), xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
        fontsize=10, fontweight="medium"
    )

    # Zoom x axis to transit duration
    if "t14" in flat_samps.data_vars:
        if flat_samps["t14"].values.ndim == 2:
            t14_med = np.median(flat_samps["t14"].values[planet_idx, :])
        else:
            t14_med = np.median(flat_samps["t14"].values)
    else:
        t14_med = 0.15

    axes[0].set_xlim(-2.5 * t14_med, 2.5 * t14_med)
    axes[0].set_ylim(1.0 - 2.5 * depth, 1.0 + 1.5 * depth)
    axes[0].set_ylabel("De-trended Relative Flux")
    axes[0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Phased MCMC Fit - Planet {planet_idx + 1}", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=9, loc="upper right")

    # ── Panel 2: Residuals in Phase Domain ──
    if transit_med is not None:
        residuals = detrended_flux - (transit_med + 1.0)
    else:
        residuals = np.zeros_like(flux)

    res_lc = lk.LightCurve(time=lc.time, flux=residuals + 1.0, flux_err=lc.flux_err)
    _, r_fold, r_fold_err = phase_fold(res_lc, period=period_med, epoch=epoch_med)

    axes[1].scatter(phase * period_med, r_fold - 1.0, s=0.8, color="gray", alpha=0.3, label="Residuals", rasterized=True)

    # Binned residuals
    bin_r_phase, bin_r_flux, bin_r_err = bin_phase_curve(phase, r_fold - 1.0, r_fold_err, n_bins=n_bins)
    valid_r = np.isfinite(bin_r_flux)
    axes[1].errorbar(
        bin_r_phase[valid_r] * period_med, bin_r_flux[valid_r], yerr=bin_r_err[valid_r],
        fmt="o", ms=4, color="#dc2626", elinewidth=0.8, capsize=2, label="Binned Residuals", zorder=10
    )

    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Residuals")
    axes[1].set_xlabel("Time since transit (days)")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_ylim(-1.5 * depth, 1.5 * depth)

    fig.tight_layout()
    return fig
