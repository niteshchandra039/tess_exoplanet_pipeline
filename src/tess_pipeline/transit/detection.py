"""
transit/detection.py — Period search orchestrator.

Runs TLS (primary) and/or BLS (triage/cross-check) on a preprocessed
light curve and returns the best-fit detection result.
"""

from __future__ import annotations

from typing import Any

from tess_pipeline.exceptions import PeriodSearchError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def search_period(
    lc: Any,
    *,
    method: str = "tls",
    period_min: float = 0.5,
    period_max: float = 100.0,
    stellar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run period search on a preprocessed light curve.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Preprocessed, stitched light curve.
    method : str
        "tls" (default), "bls", or "both".
    period_min, period_max : float
        Search range in days.
    stellar : dict | None
        Stellar parameters from ``catalogs.stellar``; used to inform
        TLS priors (stellar radius, mass).

    Returns
    -------
    dict with keys:
        period (float), epoch (float), duration_hr (float), depth (float),
        method (str), sde or snr (float),
        tls_result (if TLS run), bls_result (if BLS run)
    """
    tls_result: dict[str, Any] | None = None
    bls_result: dict[str, Any] | None = None

    if method in ("tls", "both"):
        log.info("Running TLS (period %.1f–%.1f d)", period_min, period_max)
        from tess_pipeline.transit.tls import run_tls

        tls_result = run_tls(lc, period_min=period_min, period_max=period_max, stellar=stellar)

    if method in ("bls", "both"):
        log.info("Running BLS (period %.1f–%.1f d)", period_min, period_max)
        from tess_pipeline.transit.bls import run_bls

        bls_result = run_bls(lc, period_min=period_min, period_max=period_max)

    # ── Select best result ──────────────────────────────────────────────────
    if method == "tls" and tls_result is not None:
        best = _pack(tls_result, "tls")
    elif method == "bls" and bls_result is not None:
        best = _pack(bls_result, "bls")
    elif method == "both":
        if tls_result is None and bls_result is None:
            raise PeriodSearchError("Both TLS and BLS failed to return a result.")
        # Prefer TLS (higher sensitivity) when both succeed
        best = _pack(tls_result if tls_result is not None else bls_result,
                     "tls" if tls_result is not None else "bls")
    else:
        raise PeriodSearchError(f"No period search result available (method={method!r}).")

    # Attach raw result objects for downstream diagnostics / plotting
    if tls_result is not None:
        best["tls_result"] = tls_result.get("_raw")
    if bls_result is not None:
        best["bls_result"] = bls_result.get("_raw")

    return best


def _pack(result: dict[str, Any], method: str) -> dict[str, Any]:
    """Map a TLS/BLS result dict to the canonical detection dict."""
    return {
        "period": result["period"],
        "epoch": result.get("epoch"),
        "duration_hr": result.get("duration_hr"),
        "depth": result.get("depth"),
        "method": method,
        "sde": result.get("sde"),
        "snr": result.get("snr"),
    }


def search_multiple_planets(
    lc: Any,
    *,
    method: str = "tls",
    period_min: float = 0.5,
    period_max: float = 100.0,
    stellar: dict[str, Any] | None = None,
    max_planets: int = 1,
) -> list[dict[str, Any]]:
    """
    Iteratively search for multiple transiting planets by masking found signals.
    """
    import numpy as np

    current_lc = lc.copy()
    detections = []

    for i in range(max_planets):
        if len(current_lc) < 100:
            log.info("Too few data points remaining to search for Planet %d", i + 1)
            break

        is_broad = (period_max - period_min) >= 0.5
        lc_for_search = current_lc

        if is_broad and len(current_lc) > 50_000:
            # Bin to 10-min cadence for broad search: reduces points ~5x
            # Original cadence is typically 2-min (SPOC), so bin_factor=5
            try:
                import numpy as np
                time_arr = current_lc.time.value
                time_span_days = float(np.max(time_arr) - np.min(time_arr))
                n_bins = max(1000, int(time_span_days * 24 * 6))  # 10-min bins
                lc_for_search = current_lc.bin(n_bins=n_bins)
                log.info(
                    "Binned LC from %d → %d points (10-min cadence) for broad TLS search",
                    len(current_lc), len(lc_for_search),
                )
            except Exception as bin_exc:
                log.warning("Binning failed (%s); using full-resolution LC", bin_exc)
                lc_for_search = current_lc

        log.info("Searching for Planet %d candidate...", i + 1)
        try:
            det = search_period(
                lc_for_search,
                method=method,
                period_min=period_min,
                period_max=period_max,
                stellar=stellar,
            )
        except Exception as exc:
            log.warning("Planet %d search failed: %s", i + 1, exc)
            break

        sde = det.get("sde", 0.0)
        snr = det.get("snr", 0.0)
        sig = max(sde, snr)

        # For planet 2+, enforce SDE/SNR threshold of 6.0
        if len(detections) > 0 and sig < 6.0:
            log.info("Signal significance (%.2f) below threshold of 6.0. Stopping search.", sig)
            break

        detections.append(det)
        log.info(
            "Planet %d found: Period = %.6f d, epoch = %.4f, SDE/SNR = %.2f",
            i + 1,
            det["period"],
            det["epoch"],
            sig,
        )

        if i == max_planets - 1:
            break

        # Mask the found transit in current_lc
        period = det["period"]
        t0 = det["epoch"]
        duration_days = (det["duration_hr"] or 3.0) / 24.0

        times = current_lc.time.value
        phase = (times - t0 + 0.5 * period) % period - 0.5 * period
        in_transit = np.abs(phase) < (0.75 * duration_days)

        current_lc = current_lc[~in_transit]

    return detections
