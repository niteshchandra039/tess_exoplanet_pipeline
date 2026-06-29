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
