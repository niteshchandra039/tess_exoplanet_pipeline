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


def optimized_final_T0_fit(signal, depth, t, y, dy, period, T0_fit_margin, show_progress_bar, verbose):
    """
    Optimized version of transitleastsquares.stats.final_T0_fit
    that uses a coarse-to-fine search grid to find T0 in ~100x fewer steps,
    avoiding redundant sorting and scaling with total number of data points.
    """
    import numpy as np
    from transitleastsquares.core import fold
    import transitleastsquares.tls_constants as tls_constants

    dur = len(signal)
    scale = tls_constants.SIGNAL_DEPTH / (1.0 - depth)
    signal = 1.0 - ((1.0 - signal) / scale)
    
    # Calculate median cadence
    cadence = float(np.median(np.diff(t)))
    transit_dur_days = dur * cadence
    
    # Coarse search step is 30% of transit duration (physically motivated, doesn't miss transit)
    coarse_step = 0.3 * transit_dur_days
    
    n_coarse = int(np.ceil(period / coarse_step))
    n_coarse = max(100, min(3000, n_coarse))
    
    T0_coarse = np.linspace(
        start=np.min(t), stop=np.min(t) + period, num=n_coarse
    )
    
    if verbose:
        log.info("Optimized T0 search: running coarse pass with %d points...", n_coarse)

    residuals_lowest = float("inf")
    T0_best_coarse = 0.0
    signal_ootr = np.ones(len(y[dur:]))

    for Tx in T0_coarse:
        phases = fold(time=t, period=period, T0=Tx)
        sort_index = np.argsort(phases, kind="mergesort")
        flux = y[sort_index]
        dy_sorted = dy[sort_index]

        roll_cadences = int(dur / 2) + 1
        flux = np.concatenate([flux[-roll_cadences:], flux[:-roll_cadences]])
        dy_sorted = np.concatenate([dy_sorted[-roll_cadences:], dy_sorted[:-roll_cadences]])

        residuals_intransit = np.sum((flux[:dur] - signal) ** 2 / dy_sorted[:dur] ** 2)
        residuals_ootr = np.sum((flux[dur:] - signal_ootr) ** 2 / dy_sorted[dur:] ** 2)
        residuals_total = residuals_intransit + residuals_ootr

        if residuals_total < residuals_lowest:
            residuals_lowest = residuals_total
            T0_best_coarse = Tx

    # Fine search around the best coarse T0
    fine_width = period / n_coarse
    n_fine = 300
    T0_fine = np.linspace(
        start=T0_best_coarse - fine_width, stop=T0_best_coarse + fine_width, num=n_fine
    )
    
    if verbose:
        log.info("Optimized T0 search: running fine pass with %d points around T0=%.4f...", n_fine, T0_best_coarse)

    T0_best_fine = T0_best_coarse
    for Tx in T0_fine:
        phases = fold(time=t, period=period, T0=Tx)
        sort_index = np.argsort(phases, kind="mergesort")
        flux = y[sort_index]
        dy_sorted = dy[sort_index]

        roll_cadences = int(dur / 2) + 1
        flux = np.concatenate([flux[-roll_cadences:], flux[:-roll_cadences]])
        dy_sorted = np.concatenate([dy_sorted[-roll_cadences:], dy_sorted[:-roll_cadences]])

        residuals_intransit = np.sum((flux[:dur] - signal) ** 2 / dy_sorted[:dur] ** 2)
        residuals_ootr = np.sum((flux[dur:] - signal_ootr) ** 2 / dy_sorted[dur:] ** 2)
        residuals_total = residuals_intransit + residuals_ootr

        if residuals_total < residuals_lowest:
            residuals_lowest = residuals_total
            T0_best_fine = Tx

    return T0_best_fine


def run_tls(
    lc: Any,
    *,
    period_min: float = 0.5,
    period_max: float = 100.0,
    stellar: dict[str, Any] | None = None,
    oversampling_factor: int | None = None,
    duration_grid_step: float = 1.1,
    coarse_search: bool = False,
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
    coarse_search : bool
        If True, search exactly 500 periods (highly optimized broad pass).

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
        "duration_grid_step": duration_grid_step,
    }

    is_narrow = False
    if coarse_search:
        # Create a custom coarse search grid with exactly 500 log-spaced periods
        tls_kwargs["period_grid"] = np.geomspace(period_min, period_max, num=500)
        # Still enforce minimum transits
        tls_kwargs["n_transits_min"] = 2
    else:
        is_narrow = (period_max - period_min) < 0.5
        if is_narrow:
            # Narrow search (archive period refinement): compute dense grid
            time_span = float(np.max(time) - np.min(time))
            p_mid = 0.5 * (period_min + period_max)
            half_w = 0.5 * (period_max - period_min)
            required_osf = int(1.5 * (p_mid ** 2) / (half_w * max(1.0, time_span)))
            tls_kwargs["oversampling_factor"] = max(3, min(300, required_osf))
        else:
            # Broad search (fallback): osf=1 = minimum period resolution
            tls_kwargs["oversampling_factor"] = oversampling_factor if oversampling_factor is not None else 1
            tls_kwargs["n_transits_min"] = 2

    log.debug("TLS search mode: %s (osf=%s, dur_step=%.2f)",
              "narrow" if is_narrow else "broad",
              tls_kwargs.get("oversampling_factor"), duration_grid_step)

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

    original_period_grid = None
    if coarse_search:
        import transitleastsquares.main as tls_main
        original_period_grid = tls_main.period_grid

        def mock_period_grid(*args, **kwargs):
            p_min = kwargs.get("period_min", period_min)
            p_max = kwargs.get("period_max", period_max)
            return np.geomspace(p_min, p_max, num=500)

        tls_main.period_grid = mock_period_grid

    original_final_T0_fit = None
    try:
        # Override the transitleastsquares default behavior which silently ignores 
        # narrow period ranges and falls back to a full search grid when the number of 
        # period points is small (< 100).
        import transitleastsquares.tls_constants as tls_constants
        tls_constants.MINIMUM_PERIOD_GRID_SIZE = 0

        # Monkeypatch the extremely slow final_T0_fit with our optimized coarse-to-fine search
        # We must patch both stats (where it is defined) and main (where it is imported/used)
        import transitleastsquares.stats as tls_stats
        import transitleastsquares.main as tls_main
        original_final_T0_fit = tls_stats.final_T0_fit
        tls_stats.final_T0_fit = optimized_final_T0_fit
        tls_main.final_T0_fit = optimized_final_T0_fit

        # Remove period_grid from kwargs if present to prevent TLS warning
        power_kwargs = tls_kwargs.copy()
        power_kwargs.pop("period_grid", None)

        model = transitleastsquares(time, flux, flux_err)
        results = model.power(**power_kwargs)
    except Exception as exc:
        raise PeriodSearchError(f"TLS failed: {exc}") from exc
    finally:
        if original_final_T0_fit is not None:
            import transitleastsquares.stats as tls_stats
            import transitleastsquares.main as tls_main
            tls_stats.final_T0_fit = original_final_T0_fit
            tls_main.final_T0_fit = original_final_T0_fit
        if original_period_grid is not None:
            import transitleastsquares.main as tls_main
            tls_main.period_grid = original_period_grid

    period = float(results.period)
    epoch = float(results.T0)
    duration_hr = float(results.duration) * 24.0   # days → hours
    depth = float(results.depth)
    if depth > 0.5:
        depth = 1.0 - depth
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
