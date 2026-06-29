"""
transit/phasefold.py — Phase-folded light curve generation and model overlays.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def phase_fold(
    lc: Any,
    period: float,
    epoch: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve at *period*.

    Parameters
    ----------
    lc : lightkurve.LightCurve
    period : float
        Orbital period in days.
    epoch : float | None
        Transit mid-time. If None, uses lc.time.value[0].

    Returns
    -------
    phase : np.ndarray   shape (N,), values in [-0.5, 0.5)
    flux  : np.ndarray   shape (N,)
    flux_err : np.ndarray  shape (N,) or zeros
    """
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    flux_err = (
        np.asarray(lc.flux_err.value, dtype=float)
        if lc.flux_err is not None
        else np.zeros_like(flux)
    )

    t0 = epoch if epoch is not None else time[0]
    phase = ((time - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0   # fold to [-0.5, 0.5)

    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx], flux_err[sort_idx]


def bin_phase_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a phase-folded light curve into *n_bins* equal-width phase bins.

    Returns
    -------
    bin_phase, bin_flux, bin_flux_err
    """
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_flux = np.full(n_bins, np.nan)
    bin_err = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        w = 1.0 / flux_err[mask] ** 2 if np.all(flux_err[mask] > 0) else np.ones(mask.sum())
        bin_flux[i] = np.average(flux[mask], weights=w)
        bin_err[i] = 1.0 / np.sqrt(np.sum(w))

    return bin_centers, bin_flux, bin_err
