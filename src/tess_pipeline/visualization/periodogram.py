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
    tls_result : transitleastsquares result object or dict
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if isinstance(tls_result, dict):
        raw_obj = tls_result.get("tls_result_broad") or tls_result.get("tls_result")
        if raw_obj is not None and not isinstance(raw_obj, str):
            periods = np.asarray(raw_obj.periods)
            power = np.asarray(raw_obj.power)
            best_period = float(raw_obj.period)
            sde = float(raw_obj.SDE)
        else:
            p_list = tls_result.get("tls_periods_broad") or tls_result.get("tls_periods")
            pow_list = tls_result.get("tls_power_broad") or tls_result.get("tls_power")
            if p_list is not None and pow_list is not None:
                periods = np.asarray(p_list)
                power = np.asarray(pow_list)
            else:
                periods = np.array([])
                power = np.array([])
            best_period = float(tls_result.get("period", 0.0))
            sde = float(tls_result.get("sde", 0.0))
    else:
        periods = np.asarray(tls_result.periods)
        power = np.asarray(tls_result.power)
        best_period = float(tls_result.period)
        sde = float(tls_result.SDE)

    if len(periods) > 0 and len(power) > 0:
        ax.plot(periods, power, color="steelblue", linewidth=0.8)
    best_period = float(best_period)
    ax.axvline(best_period, color="red", linestyle="--", linewidth=1.5,
               label=f"Best period: {best_period:.5f} d")

    # Mark harmonics
    for n in (2, 3):
        ax.axvline(best_period / n, color="orange", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(best_period * n, color="orange", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("TLS Power (SDE)")
    ax.set_title(f"Transit Least Squares Periodogram  |  SDE = {sde:.2f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_bls_periodogram(bls_result: Any) -> plt.Figure:
    """
    Plot the BLS power spectrum (astropy BoxLeastSquares result or dict).
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if isinstance(bls_result, dict):
        raw_obj = bls_result.get("bls_result_broad") or bls_result.get("bls_result")
        if raw_obj is not None and not isinstance(raw_obj, str):
            periods = np.asarray(raw_obj.period)
            power = np.asarray(raw_obj.power)
            best_idx = int(np.argmax(power))
            best_period = float(periods[best_idx])
        else:
            p_list = bls_result.get("bls_periods_broad") or bls_result.get("bls_periods")
            pow_list = bls_result.get("bls_power_broad") or bls_result.get("bls_power")
            if p_list is not None and pow_list is not None:
                periods = np.asarray(p_list)
                power = np.asarray(pow_list)
                best_idx = int(np.argmax(power))
                best_period = float(periods[best_idx])
            else:
                periods = np.array([])
                power = np.array([])
                best_period = float(bls_result.get("period", 0.0))
    else:
        periods = np.asarray(bls_result.period)
        power = np.asarray(bls_result.power)
        best_idx = int(np.argmax(power))
        best_period = float(periods[best_idx])

    if len(periods) > 0 and len(power) > 0:
        ax.plot(periods, power, color="darkorange", linewidth=0.8)
    best_period = float(best_period)
    ax.axvline(best_period, color="red", linestyle="--", linewidth=1.5,
               label=f"Best period: {best_period:.5f} d")

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS Power")
    ax.set_title(f"Box Least Squares Periodogram  |  Best P = {best_period:.5f} d")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
