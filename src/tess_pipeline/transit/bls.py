"""
transit/bls.py — Box Least Squares period search.

Wraps the astropy BLS implementation. Faster than TLS but less sensitive
for small planets. Used for quick triage and cross-check.
"""

from __future__ import annotations

from typing import Any

from tess_pipeline.exceptions import PeriodSearchError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def run_bls(
    lc: Any,
    *,
    period_min: float = 0.5,
    period_max: float = 100.0,
    duration_grid: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3),
) -> dict[str, Any]:
    """
    Run Box Least Squares on *lc*.

    Parameters
    ----------
    lc : lightkurve.LightCurve
    period_min, period_max : float
        Search range in days.
    duration_grid : tuple[float, ...]
        Transit duration grid in days.

    Returns
    -------
    dict with keys:
        period (float), epoch (float), duration_hr (float),
        depth (float), power (float), snr (float), _raw (BLS result object)
    """
    try:
        from astropy.timeseries import BoxLeastSquares
        import numpy as np
    except ImportError as exc:
        raise PeriodSearchError("astropy is required for BLS search") from exc

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    flux_err = (
        np.asarray(lc.flux_err.value, dtype=float) if lc.flux_err is not None else None
    )

    try:
        model = BoxLeastSquares(time, flux, dy=flux_err)
        period_grid = np.linspace(period_min, period_max, 10_000)
        bls_result = model.power(period_grid, duration_grid)
    except Exception as exc:
        raise PeriodSearchError(f"BLS failed: {exc}") from exc

    best_idx = int(np.argmax(bls_result.power))
    period = float(bls_result.period[best_idx])
    epoch = float(bls_result.transit_time[best_idx])
    duration_hr = float(bls_result.duration[best_idx]) * 24.0
    depth = float(bls_result.depth[best_idx])
    power = float(bls_result.power[best_idx])

    # Signal-to-noise proxy: power / median(power)
    median_power = float(np.median(bls_result.power))
    snr = power / median_power if median_power > 0 else 0.0

    log.info(
        "BLS: period=%.6f d, epoch=%.4f, duration=%.2f h, depth=%.6f, power=%.2f",
        period, epoch, duration_hr, depth, power,
    )

    return {
        "period": period,
        "epoch": epoch,
        "duration_hr": duration_hr,
        "depth": depth,
        "power": power,
        "snr": snr,
        "_raw": bls_result,
    }
