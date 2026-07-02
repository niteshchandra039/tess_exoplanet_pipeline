"""
visualization/periodogram.py — TLS and BLS periodogram plots.
"""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np


def plot_tls_periodogram(detections: Any, tic_id: str = "", sectors_str: str = "", mode: str = "auto") -> plt.Figure:
    """
    Plot the TLS power spectrum (supports multi-planet detection panels).
    
    Parameters
    ----------
    detections : Any
        A single detection dict or list of detection dicts/objects.
    tic_id : str
    sectors_str : str
    mode : str
        "auto" (default: prefers coarse/broad if available, else fine),
        "coarse" (forces plotting the coarse/broad search),
        "fine" (forces plotting the refined/narrow search).
    """
    if not isinstance(detections, list):
        # Handle dict or other objects
        if isinstance(detections, dict) and "detections" in detections:
            detections = detections["detections"]
        else:
            detections = [detections]

    n_panels = len(detections)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), sharex=False, squeeze=False)

    for idx, det in enumerate(detections):
        ax = axes[idx, 0]

        if isinstance(det, dict) and "method" in det:
            if mode == "coarse":
                raw_obj = det.get("tls_result_broad") or det.get("tls_result")
            elif mode == "fine":
                raw_obj = det.get("tls_result")
            else:
                raw_obj = det.get("tls_result_broad") or det.get("tls_result")

            if raw_obj is not None and not isinstance(raw_obj, str):
                periods = np.asarray(raw_obj.periods)
                power = np.asarray(raw_obj.power)
                best_period = float(raw_obj.period)
                sde = float(raw_obj.SDE)
            else:
                if mode == "coarse":
                    p_list = det.get("tls_periods_broad") or det.get("tls_periods")
                    pow_list = det.get("tls_power_broad") or det.get("tls_power")
                elif mode == "fine":
                    p_list = det.get("tls_periods")
                    pow_list = det.get("tls_power")
                else:
                    p_list = det.get("tls_periods_broad") or det.get("tls_periods")
                    pow_list = det.get("tls_power_broad") or det.get("tls_power")

                if p_list is not None and pow_list is not None:
                    periods = np.asarray(p_list)
                    power = np.asarray(pow_list)
                else:
                    periods = np.array([])
                    power = np.array([])
                best_period = float(det.get("period", 0.0))
                sde = float(det.get("sde", 0.0))
        else:
            periods = np.asarray(det.periods)
            power = np.asarray(det.power)
            best_period = float(det.period)
            sde = float(det.SDE)

        if len(periods) > 0 and len(power) > 0:
            ax.plot(periods, power, color="#2563eb", linewidth=0.8)
            ax.set_xlim(np.min(periods), np.max(periods))
            ax.set_ylim(0, np.max(power) * 1.1)

        best_period = float(best_period)
        ax.axvline(best_period, color="#dc2626", linestyle="--", linewidth=1.5,
                   label=f"Best period: {best_period:.5f} d")

        # Mark harmonics
        for n in (2, 3):
            ax.axvline(best_period / n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.axvline(best_period * n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)

        ax.set_ylabel("TLS Power (SDE)")
        ax.legend(fontsize=9, loc="upper right")
        
        title_suffix = f" (Planet {idx + 1} Search)" if len(detections) > 1 else ""
        mode_str = " Coarse" if mode == "coarse" else (" Fine" if mode == "fine" else "")
        ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | TLS{mode_str} Periodogram{title_suffix} (SDE = {sde:.2f})", fontsize=10, fontweight="bold")

    axes[-1, 0].set_xlabel("Period (days)")
    fig.tight_layout()
    return fig


def plot_bls_periodogram(detections: Any, tic_id: str = "", sectors_str: str = "", mode: str = "auto") -> plt.Figure:
    """
    Plot the BLS power spectrum (supports multi-planet detection panels).
    
    Parameters
    ----------
    detections : Any
    tic_id : str
    sectors_str : str
    mode : str
        "auto" (default: prefers coarse/broad if available, else fine),
        "coarse" (forces plotting the coarse/broad search),
        "fine" (forces plotting the refined/narrow search).
    """
    if not isinstance(detections, list):
        if isinstance(detections, dict) and "detections" in detections:
            detections = detections["detections"]
        else:
            detections = [detections]

    n_panels = len(detections)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), sharex=False, squeeze=False)

    for idx, det in enumerate(detections):
        ax = axes[idx, 0]

        if isinstance(det, dict) and "method" in det:
            if mode == "coarse":
                raw_obj = det.get("bls_result_broad") or det.get("bls_result")
            elif mode == "fine":
                raw_obj = det.get("bls_result")
            else:
                raw_obj = det.get("bls_result_broad") or det.get("bls_result")

            if raw_obj is not None and not isinstance(raw_obj, str):
                periods = np.asarray(raw_obj.period)
                power = np.asarray(raw_obj.power)
                best_idx = int(np.argmax(power))
                best_period = float(periods[best_idx])
                snr = float(det.get("snr", 0.0))
            else:
                if mode == "coarse":
                    p_list = det.get("bls_periods_broad") or det.get("bls_periods")
                    pow_list = det.get("bls_power_broad") or det.get("bls_power")
                elif mode == "fine":
                    p_list = det.get("bls_periods")
                    pow_list = det.get("bls_power")
                else:
                    p_list = det.get("bls_periods_broad") or det.get("bls_periods")
                    pow_list = det.get("bls_power_broad") or det.get("bls_power")

                if p_list is not None and pow_list is not None:
                    periods = np.asarray(p_list)
                    power = np.asarray(pow_list)
                    best_idx = int(np.argmax(power))
                    best_period = float(periods[best_idx])
                else:
                    periods = np.array([])
                    power = np.array([])
                    best_period = float(det.get("period", 0.0))
                snr = float(det.get("snr", 0.0))
        else:
            periods = np.asarray(det.period)
            power = np.asarray(det.power)
            best_idx = int(np.argmax(power))
            best_period = float(periods[best_idx])
            snr = float(getattr(det, "snr", 0.0))

        if len(periods) > 0 and len(power) > 0:
            ax.plot(periods, power, color="#d97706", linewidth=0.8)
            ax.set_xlim(np.min(periods), np.max(periods))
            ax.set_ylim(0, np.max(power) * 1.1)

        best_period = float(best_period)
        ax.axvline(best_period, color="#dc2626", linestyle="--", linewidth=1.5,
                   label=f"Best period: {best_period:.5f} d")

        ax.set_ylabel("BLS Power")
        ax.legend(fontsize=9, loc="upper right")
        
        title_suffix = f" (Planet {idx + 1} Search)" if len(detections) > 1 else ""
        mode_str = " Coarse" if mode == "coarse" else (" Fine" if mode == "fine" else "")
        ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | BLS{mode_str} Periodogram{title_suffix} (SNR = {snr:.2f})", fontsize=10, fontweight="bold")

    axes[-1, 0].set_xlabel("Period (days)")
    fig.tight_layout()
    return fig
