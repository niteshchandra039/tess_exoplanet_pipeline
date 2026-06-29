"""
transit/tls.py — Transit Least Squares period search.

Wraps the ``transitleastsquares`` package (Hippke et al. 2019).
TLS uses realistic limb-darkened transit templates and achieves
~10–17% higher recovery rates for small planets vs BLS.

Reference: Hippke et al. 2019, A&A 623, A39
           https://doi.org/10.1051/0004-6361/201834672
"""

from __future__ import annotations

from typing import Any

from tess_pipeline.exceptions import PeriodSearchError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def run_tls(
    lc: Any,
    *,
    period_min: float = 0.5,
    period_max: float = 100.0,
    stellar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run Transit Least Squares on *lc*.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Preprocessed light curve (normalised flux ≈ 1).
    period_min, period_max : float
        Search range in days.
    stellar : dict | None
        Stellar parameters; provides R★ and M★ priors for TLS templates.

    Returns
    -------
    dict with keys:
        period (float), epoch (float), duration_hr (float),
        depth (float), sde (float), snr (float), _raw (TLS result object)
    """
    try:
        from transitleastsquares import transitleastsquares, catalog_info
    except ImportError as exc:
        raise PeriodSearchError(
            "transitleastsquares is required for TLS search. "
            "Install with: pip install transitleastsquares"
        ) from exc

    import numpy as np

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    flux_err = np.asarray(lc.flux_err.value, dtype=float) if lc.flux_err is not None else None

    # ── Build kwargs for TLS ──────────────────────────────────────────────────
    tls_kwargs: dict[str, Any] = {
        "period_min": period_min,
        "period_max": period_max,
    }

    if stellar is not None:
        r_star = stellar.get("r_star")
        m_star = stellar.get("m_star")
        r_star_err = stellar.get("r_star_err")
        m_star_err = stellar.get("m_star_err")
        if r_star is not None:
            tls_kwargs["R_star"] = r_star
            tls_kwargs["R_star_min"] = max(0.01, r_star - 3 * (r_star_err or 0.1))
            tls_kwargs["R_star_max"] = r_star + 3 * (r_star_err or 0.1)
        if m_star is not None:
            tls_kwargs["M_star"] = m_star
            tls_kwargs["M_star_min"] = max(0.01, m_star - 3 * (m_star_err or 0.1))
            tls_kwargs["M_star_max"] = m_star + 3 * (m_star_err or 0.1)

    log.debug("TLS kwargs: %s", {k: v for k, v in tls_kwargs.items() if "_raw" not in k})

    try:
        model = transitleastsquares(time, flux, flux_err)
        results = model.power(**tls_kwargs)
    except Exception as exc:
        raise PeriodSearchError(f"TLS failed: {exc}") from exc

    period = float(results.period)
    epoch = float(results.T0)
    duration_hr = float(results.duration) * 24.0   # days → hours
    depth = float(results.depth)
    sde = float(results.SDE)
    snr = float(results.snr)

    log.info(
        "TLS: period=%.6f d, epoch=%.4f, duration=%.2f h, depth=%.6f, SDE=%.2f",
        period, epoch, duration_hr, depth, sde,
    )

    return {
        "period": period,
        "epoch": epoch,
        "duration_hr": duration_hr,
        "depth": depth,
        "sde": sde,
        "snr": snr,
        "_raw": results,
    }
