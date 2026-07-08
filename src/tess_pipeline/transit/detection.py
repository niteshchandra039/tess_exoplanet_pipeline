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
    is_archive: bool = False,
    coarse_search: bool = False,
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
    is_archive : bool
        Whether this is a refinement of a known archive planet.
    coarse_search : bool
        If True, search exactly 500 periods (highly optimized broad pass).

    Returns
    -------
    dict with keys:
        period (float), epoch (float), duration_hr (float), depth (float),
        method (str), sde or snr (float), fap (float), confidence (str),
        existence_probability_search (float),
        tls_result (if TLS run), bls_result (if BLS run)
    """
    tls_result: dict[str, Any] | None = None
    bls_result: dict[str, Any] | None = None

    if method in ("tls", "both"):
        log.info("Running TLS (period %.1f–%.1f d)", period_min, period_max)
        from tess_pipeline.transit.tls import run_tls

        tls_result = run_tls(
            lc,
            period_min=period_min,
            period_max=period_max,
            stellar=stellar,
            coarse_search=coarse_search,
        )

    if method in ("bls", "both"):
        log.info("Running BLS (period %.1f–%.1f d)", period_min, period_max)
        from tess_pipeline.transit.bls import run_bls

        bls_result = run_bls(lc, period_min=period_min, period_max=period_max)

    # ── Select best result ──────────────────────────────────────────────────
    if method == "tls" and tls_result is not None:
        best = _pack(tls_result, "tls", is_archive=is_archive)
    elif method == "bls" and bls_result is not None:
        best = _pack(bls_result, "bls", is_archive=is_archive)
    elif method == "both":
        if tls_result is None and bls_result is None:
            raise PeriodSearchError("Both TLS and BLS failed to return a result.")
        # Prefer TLS (higher sensitivity) when both succeed
        best = _pack(tls_result if tls_result is not None else bls_result,
                     "tls" if tls_result is not None else "bls",
                     is_archive=is_archive)
    else:
        raise PeriodSearchError(f"No period search result available (method={method!r}).")

    # Attach raw result objects for downstream diagnostics / plotting
    if tls_result is not None:
        raw_tls = tls_result.get("_raw")
        best["tls_result"] = raw_tls
        if raw_tls is not None:
            if hasattr(raw_tls, "periods"):
                best["tls_periods"] = list(raw_tls.periods)
            if hasattr(raw_tls, "power"):
                best["tls_power"] = list(raw_tls.power)
    if bls_result is not None:
        raw_bls = bls_result.get("_raw")
        best["bls_result"] = raw_bls
        if raw_bls is not None:
            if hasattr(raw_bls, "period"):
                best["bls_periods"] = list(raw_bls.period)
            if hasattr(raw_bls, "power"):
                best["bls_power"] = list(raw_bls.power)

    return best


def _pack(result: dict[str, Any], method: str, is_archive: bool = False) -> dict[str, Any]:
    """Map a TLS/BLS result dict to the canonical detection dict with FAP and confidence."""
    d = {
        "period": result["period"],
        "epoch": result.get("epoch"),
        "duration_hr": result.get("duration_hr"),
        "depth": result.get("depth"),
        "method": method,
        "sde": result.get("sde"),
        "snr": result.get("snr"),
    }

    if is_archive:
        d["fap"] = 0.0
        d["confidence"] = "Confirmed"
        d["existence_probability_search"] = 1.0
        return d

    # Calculate FAP and confidence label if it's TLS
    if method == "tls" and result.get("sde") is not None:
        import numpy as np
        sde = float(result["sde"])
        raw = result.get("_raw")
        n_trials = len(raw.periods) if (raw is not None and hasattr(raw, "periods")) else 1000

        # SDE-to-FAP conversion
        if sde > 100:
            fap = 0.0
        else:
            fap = float(1.0 - (1.0 - np.exp(-sde))**n_trials)
            fap = max(0.0, min(1.0, fap))

        d["fap"] = fap

        # Assign confidence category and existence probability consistently
        d["existence_probability_search"] = float(1.0 - fap)

        if fap > 0.50:
            d["confidence"] = "Low"
        elif fap > 0.05:
            d["confidence"] = "Moderate"
        else:
            d["confidence"] = "High"
    else:
        d["fap"] = None
        d["confidence"] = "Low"
        d["existence_probability_search"] = 0.5

    return d



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
            # Downsample to 10-min cadence for broad search (reduces points ~5x)
            # This is extremely fast, preserves signal, and avoids NaN empty-bin bugs in gaps.
            lc_for_search = current_lc[::5]
            log.info(
                "Downsampled LC from %d → %d points (10-min cadence) for broad TLS search",
                len(current_lc), len(lc_for_search),
            )


        # Cap period_max: require at least 3 complete transits in the baseline
        import numpy as np
        time_arr = lc_for_search.time.value
        t_baseline = float(np.max(time_arr) - np.min(time_arr))
        effective_period_max = min(period_max, t_baseline / 3.0)
        if effective_period_max < period_min:
            effective_period_max = period_max  # fallback if baseline too short
        if effective_period_max < period_max:
            log.info(
                "Capping period_max %.1f → %.1f d (3-transit requirement over %.0f d baseline)",
                period_max, effective_period_max, t_baseline,
            )

        log.info("Searching for Planet %d candidate (coarse search)...", i + 1)
        try:
            det = search_period(
                lc_for_search,
                method=method,
                period_min=period_min,
                period_max=effective_period_max,
                stellar=stellar,
                coarse_search=True,
            )
        except Exception as exc:
            log.warning("Planet %d coarse search failed: %s", i + 1, exc)
            break

        coarse_sde = det.get("sde", 0.0)
        coarse_snr = det.get("snr", 0.0)
        coarse_sig = max(coarse_sde, coarse_snr)

        # Stop early if coarse search finds nothing of interest (threshold of 4.0)
        if coarse_sig < 4.0:
            log.info("Planet %d coarse SDE/SNR (%.2f) below threshold of 4.0. Stopping search.", i + 1, coarse_sig)
            break

        # Zoom in and run high-resolution narrow search on full-resolution data
        # while checking sub-harmonics / aliases to robustly avoid harmonics
        coarse_period = det["period"]
        
        log.info(
            "Significant peak found (coarse SDE/SNR = %.2f at %.4f d). Checking sub-harmonics for robustness...",
            coarse_sig, coarse_period,
        )

        fractions = [1/3, 1/2, 1, 2, 3]
        alias_results = []
        
        for f in fractions:
            test_p = coarse_period * f
            if test_p < period_min or test_p > effective_period_max:
                continue
                
            delta_p = max(0.01, 0.02 * test_p)
            narrow_min = max(period_min, test_p - delta_p)
            narrow_max = min(effective_period_max, test_p + delta_p)
            
            try:
                ref_det = search_period(
                    current_lc,
                    method=method,
                    period_min=narrow_min,
                    period_max=narrow_max,
                    stellar=stellar,
                    coarse_search=False,
                )
                sig = max(ref_det.get("sde", 0.0), ref_det.get("snr", 0.0))
                if sig > 0:
                    alias_results.append({
                        "f": f,
                        "det": ref_det,
                        "sig": sig,
                        "period": ref_det["period"]
                    })
            except Exception as exc:
                log.warning("Planet %d harmonic check failed for f=%.2f: %s", i + 1, f, exc)
                
        if not alias_results:
            log.warning("No valid refined results for Planet %d", i + 1)
            break
            
        # Find the max signal across all tested fractions
        # We heavily trust the SDE/SNR to tell the truth. TLS correctly penalizes 
        # multiple/sub-harmonics. Instead of biasing towards shorter periods, we simply 
        # select the one that yields the absolute highest signal, as that's mathematically
        # the best fit for true physical transits without ghost epochs.
        max_sig = max(res["sig"] for res in alias_results)
        
        chosen_res = None
        for res in alias_results:
             if res["sig"] == max_sig:
                 chosen_res = res
                 break
                
        refined_sig = chosen_res["sig"]
        refined_det = chosen_res["det"]
        f_chosen = chosen_res["f"]

        log.info(
            "Harmonic resolution: selected fraction %.2f (refined period=%.4f d, SDE/SNR=%.2f, max_sig was %.2f)",
            f_chosen, chosen_res["period"], refined_sig, max_sig
        )
        print(
            f"Planet {i+1} derived robust period: {chosen_res['period']:.5f} days "
            f"(SNR: {refined_sig:.2f}, via fraction {f_chosen:.2f} of initial peak {coarse_period:.5f}d)"
        )

        # Save the broad/coarse search raw and list results so they can be plotted
        broad_tls = det.get("tls_result")
        broad_bls = det.get("bls_result")
        broad_tls_periods = det.get("tls_periods")
        broad_tls_power = det.get("tls_power")
        broad_bls_periods = det.get("bls_periods")
        broad_bls_power = det.get("bls_power")

        det = refined_det

        if broad_tls is not None:
            det["tls_result_broad"] = broad_tls
        if broad_bls is not None:
            det["bls_result_broad"] = broad_bls
        if broad_tls_periods is not None:
            det["tls_periods_broad"] = broad_tls_periods
        if broad_tls_power is not None:
            det["tls_power_broad"] = broad_tls_power
        if broad_bls_periods is not None:
            det["bls_periods_broad"] = broad_bls_periods
        if broad_bls_power is not None:
            det["bls_power_broad"] = broad_bls_power

        # For subsequent planets (Planet 2+), enforce SDE/SNR threshold of 6.0
        if len(detections) > 0 and refined_sig < 6.0:
            log.info("Planet %d refined SDE/SNR (%.2f) below threshold of 6.0. Stopping search.", i + 1, refined_sig)
            break

        detections.append(det)
        log.info(
            "Planet %d found: Period = %.6f d, epoch = %.4f, SDE/SNR = %.2f",
            i + 1,
            det["period"],
            det["epoch"],
            refined_sig,
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
