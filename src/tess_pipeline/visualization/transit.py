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
) -> plt.Figure:
    """
    Plot a phase-folded light curve with an optional analytic model overlay.

    Parameters
    ----------
    lc : lightkurve.LightCurve
    period : float
    epoch : float | None
    model : dict | None
        Dict with 'time', 'flux_model' arrays for model overlay.
    n_bins : int
        Number of phase bins.
    """
    phase, flux, flux_err = phase_fold(lc, period, epoch)
    bin_phase, bin_flux, bin_err = bin_phase_curve(phase, flux, flux_err, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw scatter
    ax.scatter(phase, flux, s=1.0, color="steelblue", alpha=0.3, rasterized=True, label="Data")

    # Binned
    valid = np.isfinite(bin_flux)
    ax.errorbar(
        bin_phase[valid], bin_flux[valid], yerr=bin_err[valid],
        fmt="o", ms=3, color="navy", elinewidth=0.8, capsize=2, label="Binned",
    )

    # Model overlay
    if model is not None and model.get("time") is not None:
        model_flux = model.get("transit_model") if model.get("transit_model") is not None else model.get("flux_model")
        if model_flux is not None:
            m_phase = (
                (np.asarray(model["time"]) - (epoch or lc.time.value[0])) / period
            ) % 1.0
            m_phase[m_phase >= 0.5] -= 1.0
            sort_idx = np.argsort(m_phase)
            ax.plot(
                m_phase[sort_idx], np.asarray(model_flux)[sort_idx],
                color="red", linewidth=1.5, zorder=5, label="Model",
            )

    ax.set_xlabel(f"Phase (P = {period:.5f} d)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title("Phase-Folded Transit")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_batman_overlay(
    lc: Any,
    period: float,
    epoch: float | None,
    batman_result: dict[str, Any],
) -> plt.Figure:
    """Plot batman analytic model overlay on phase-folded data."""
    return plot_phase_curve(lc, period, epoch, model=batman_result)


def plot_bayesian_fit(
    lc: Any,
    posterior: Any,
    model_outputs: dict[str, Any] | None,
) -> plt.Figure:
    """
    Plot the Bayesian posterior median transit fit.

    Generates a 3-panel diagnostic plot:
      1. Normalized flux with GP systematics model overlaid.
      2. De-trended flux with Keplerian transit model overlaid.
      3. Residuals after subtracting the full model.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)

    # Defaults in case keys are missing
    gp_model = model_outputs.get("gp_model") if model_outputs else None
    transit_model = model_outputs.get("transit_model") if model_outputs else None
    flux_model = model_outputs.get("flux_model") if model_outputs else None

    # ── Panel 1: Data & GP Systematics ──────────────────────────────────────
    axes[0].scatter(time, flux, s=0.5, color="black", alpha=0.3, label="Data", rasterized=True)
    if gp_model is not None:
        if transit_model is not None:
            trend = flux_model - (transit_model - 1.0)
        else:
            trend = gp_model + np.median(flux)
        axes[0].plot(time, trend, color="#22c55e", linewidth=1.2, label="GP + Trend")
    axes[0].set_ylabel("Relative Flux")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title("Bayesian GP Detrending & Transit Fit", fontsize=11, fontweight="bold")

    # ── Panel 2: De-trended Data & Transit Model ────────────────────────────
    if gp_model is not None:
        detrended_flux = flux - gp_model
    else:
        detrended_flux = flux
    axes[1].scatter(time, detrended_flux, s=0.5, color="black", alpha=0.3, label="De-trended Data", rasterized=True)
    if transit_model is not None:
        axes[1].plot(time, transit_model, color="#2563eb", linewidth=1.2, label="Transit Model")
    axes[1].set_ylabel("De-trended Flux")
    axes[1].legend(fontsize=9, loc="upper right")

    # ── Panel 3: Residuals ──────────────────────────────────────────────────
    if flux_model is not None:
        residuals = flux - flux_model
    else:
        residuals = np.zeros_like(time)
    axes[2].scatter(time, residuals, s=0.5, color="gray", alpha=0.3, label="Residuals", rasterized=True)
    axes[2].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Residuals")
    axes[2].set_xlabel("Time (BTJD)")
    axes[2].legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    return fig


def plot_mcmc_phase_curve(
    lc: Any,
    posterior: Any,
    period: float,
    epoch: float | None = None,
    n_bins: int = 50,
) -> plt.Figure:
    """
    Plot the phase-folded light curve using MCMC posterior samples.
    Shows the de-trended data (data - GP), binned data, and the posterior median
    transit model with its 68% credible interval (shaded region).
    """
    import lightkurve as lk
    from tess_pipeline.transit.phasefold import phase_fold, bin_phase_curve

    fig, ax = plt.subplots(figsize=(10, 5))

    time = np.asarray(lc.time.value)
    flux = np.asarray(lc.flux.value)
    flux_err = np.asarray(lc.flux_err.value) if lc.flux_err is not None else np.ones_like(flux) * 1e-3

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

    # De-trend the data
    detrended_flux = flux - gp_mod

    # Phase-fold the de-trended data using a temporary lightkurve LightCurve
    detrended_lc = lk.LightCurve(time=lc.time, flux=detrended_flux, flux_err=lc.flux_err)
    phase, y_fold, y_fold_err = phase_fold(
        detrended_lc,
        period=period,
        epoch=epoch,
    )

    # Plot raw scatter
    ax.scatter(phase, y_fold, s=1.0, color="black", alpha=0.25, label="Data", rasterized=True, zorder=-1000)

    # Plot binned data
    bin_phase, bin_flux, bin_err = bin_phase_curve(phase, y_fold, y_fold_err, n_bins=n_bins)
    valid = np.isfinite(bin_flux)
    ax.errorbar(
        bin_phase[valid], bin_flux[valid], yerr=bin_err[valid],
        fmt="o", ms=4, color="#16a34a", elinewidth=0.8, capsize=2, label="Binned", zorder=100
    )

    # Plot the posterior median model with 68% credible interval (16th to 84th percentile)
    if "lc_pred" in flat_samps.data_vars:
        phase_grid = np.linspace(-0.3, 0.3, 200)
        pred = np.percentile(flat_samps["lc_pred"].values, [16, 50, 84], axis=-1)

        # Plot median model
        ax.plot(phase_grid, pred[1], color="#ea580c", linewidth=2.0, label="Transit Model", zorder=1000)
        # Plot shaded region
        art = ax.fill_between(
            phase_grid, pred[0], pred[2],
            color="#ea580c", alpha=0.35, zorder=500, label="68% Credible Band"
        )
        art.set_edgecolor("none")
    elif "transit_model" in flat_samps.data_vars:
        transit_med = np.median(flat_samps["transit_model"].values, axis=-1)
        transit_lc = lk.LightCurve(time=lc.time, flux=transit_med, flux_err=lc.flux_err)
        _, t_fold, _ = phase_fold(
            transit_lc,
            period=period,
            epoch=epoch,
        )
        sort_idx = np.argsort(phase)
        ax.plot(phase[sort_idx], t_fold[sort_idx], color="#ea580c", linewidth=2.0, label="Transit Model", zorder=1000)

    # Annotate the period
    period_med = np.median(flat_samps["period"].values) if "period" in flat_samps.data_vars else period
    period_std = np.std(flat_samps["period"].values) if "period" in flat_samps.data_vars else 0.0
    txt = f"P = {period_med:.5f} ± {period_std:.5f} d"
    ax.annotate(
        txt, (0.02, 0.05), xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"),
        fontsize=10, fontweight="medium"
    )

    ax.set_xlim(-0.2, 0.2)
    # Autoscale y to focus on transit
    ax.set_ylim(np.percentile(y_fold, 1) - 0.002, np.percentile(y_fold, 99) + 0.002)
    ax.set_xlabel("Time since transit [days]")
    ax.set_ylabel("De-trended Relative Flux")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig
