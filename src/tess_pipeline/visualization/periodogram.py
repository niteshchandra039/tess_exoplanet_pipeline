"""
visualization/periodogram.py — TLS and BLS periodogram plots.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_tls_periodogram(tls_result: Any) -> plt.Figure:
    """
    Plot the TLS power spectrum.

    Parameters
    ----------
    tls_result : transitleastsquares result object
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    periods = np.asarray(tls_result.periods)
    power = np.asarray(tls_result.power)

    ax.plot(periods, power, color="steelblue", linewidth=0.8)
    best_period = float(tls_result.period)
    ax.axvline(best_period, color="red", linestyle="--", linewidth=1.5,
               label=f"Best period: {best_period:.5f} d")

    # Mark harmonics
    for n in (2, 3):
        ax.axvline(best_period / n, color="orange", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(best_period * n, color="orange", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("TLS Power (SDE)")
    ax.set_title(f"Transit Least Squares Periodogram  |  SDE = {tls_result.SDE:.2f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_bls_periodogram(bls_result: Any) -> plt.Figure:
    """
    Plot the BLS power spectrum (astropy BoxLeastSquares result).
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    periods = np.asarray(bls_result.period)
    power = np.asarray(bls_result.power)

    ax.plot(periods, power, color="darkorange", linewidth=0.8)
    best_idx = int(np.argmax(power))
    best_period = float(periods[best_idx])
    ax.axvline(best_period, color="red", linestyle="--", linewidth=1.5,
               label=f"Best period: {best_period:.5f} d")

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS Power")
    ax.set_title(f"Box Least Squares Periodogram  |  Best P = {best_period:.5f} d")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
