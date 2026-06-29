"""
data/preprocess.py — Light curve preprocessing wrappers around lightkurve.

Steps:
  1. Remove NaNs
  2. Quality bit masking (applied per sector by lightkurve)
  3. Outlier sigma clipping
  4. Stitch multi-sector collection
  5. Initial flattening / detrending (first-pass only; GP handles residuals)

Returns a single clean lightkurve.LightCurve object normalised to unit flux.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from tess_pipeline.exceptions import PreprocessingError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def preprocess(
    lc_collection: Any,
    *,
    period: float | None = None,
    epoch: float | None = None,
    transit_mask_width: float = 0.15,
    sigma_clip_lower: float = 20.0,
    sigma_clip_upper: float = 5.0,
    flatten_window_length: int = 401,
    flatten_polyorder: int = 3,
    flatten_break_tolerance: int = 5,
) -> Any:
    """
    Preprocess a ``lightkurve.LightCurveCollection`` into a single
    clean, stitched, and flattened light curve.

    Parameters
    ----------
    lc_collection : lightkurve.LightCurveCollection
    period : float | None
        Orbital period of the planet in days (optional, to construct a transit mask).
    epoch : float | None
        Transit mid-time (optional, to construct a transit mask).
    transit_mask_width : float
        Width (in days) around the transit center to mask out (default: 0.15 days).
    sigma_clip_lower, sigma_clip_upper : float
        Sigma thresholds for outlier removal.
    flatten_window_length : int
        Savitzky-Golay window length in cadences (must be odd).
    flatten_polyorder : int
        Savitzky-Golay polynomial order.
    flatten_break_tolerance : int
        Gap size (cadences) that triggers a new segment in flatten().

    Returns
    -------
    lightkurve.LightCurve
        Normalised, stitched, and flattened light curve.
    """
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise PreprocessingError("lightkurve is required for preprocessing") from exc

    if lc_collection is None or len(lc_collection) == 0:
        raise PreprocessingError("Empty light curve collection; nothing to preprocess.")

    # ── Per-sector: remove NaNs and clip outliers (ignoring transit window if mask is active) ──
    cleaned = []
    for lc in lc_collection:
        lc = lc.remove_nans()

        # Construct a transit mask to protect transit points from clipping
        transit_mask = np.zeros(len(lc), dtype=bool)
        if period is not None and epoch is not None:
            t0 = epoch
            if t0 > 2400000 and np.median(lc.time.value) < 100000:
                t0 -= 2457000.0  # Convert BJD to BTJD if needed
            phase = ((lc.time.value - t0) / period) % 1.0
            phase[phase > 0.5] -= 1.0
            transit_mask = np.abs(phase) < (transit_mask_width / period)

        try:
            if np.any(transit_mask):
                non_transit_indices = np.where(~transit_mask)[0]
                sub_lc = lc[non_transit_indices]
                _, outlier_mask = sub_lc.remove_outliers(
                    sigma_lower=sigma_clip_lower,
                    sigma_upper=sigma_clip_upper,
                    return_mask=True,
                )
                full_outlier_mask = np.zeros(len(lc), dtype=bool)
                full_outlier_mask[non_transit_indices] = outlier_mask
                lc = lc[~full_outlier_mask]
            else:
                lc = lc.remove_outliers(
                    sigma_lower=sigma_clip_lower,
                    sigma_upper=sigma_clip_upper,
                )
        except Exception as exc:
            log.warning("Outlier removal failed on sector: %s", exc)

        cleaned.append(lc)

    if not cleaned:
        raise PreprocessingError("All sectors removed after NaN/outlier cleaning.")

    # ── Stitch sectors ────────────────────────────────────────────────────────
    collection = lk.LightCurveCollection(cleaned)
    stitched = collection.stitch()
    log.debug(
        "Stitched %d sectors → %d cadences",
        len(cleaned),
        len(stitched),
    )

    # ── Ensure odd window length ──────────────────────────────────────────────
    if flatten_window_length % 2 == 0:
        flatten_window_length += 1
        log.debug("Adjusted flatten_window_length to %d (must be odd)", flatten_window_length)

    # Re-calculate transit mask on stitched light curve for flattening
    transit_mask_stitched = np.zeros(len(stitched), dtype=bool)
    if period is not None and epoch is not None:
        t0 = epoch
        if t0 > 2400000 and np.median(stitched.time.value) < 100000:
            t0 -= 2457000.0
        phase = ((stitched.time.value - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        transit_mask_stitched = np.abs(phase) < (transit_mask_width / period)

    # ── Initial flatten (first-pass detrend) with transit mask ─────────────────
    try:
        flat, trend = stitched.flatten(
            window_length=flatten_window_length,
            polyorder=flatten_polyorder,
            break_tolerance=flatten_break_tolerance,
            mask=transit_mask_stitched if np.any(transit_mask_stitched) else None,
            return_trend=True,
        )
    except Exception as exc:
        log.warning("Flattening failed (%s); using unflattened light curve", exc)
        flat = stitched

    # Final NaN pass and outlier removal after flattening
    flat = flat.remove_nans()
    try:
        if np.any(transit_mask_stitched):
            non_transit_indices = np.where(~transit_mask_stitched)[0]
            sub_flat = flat[non_transit_indices]
            _, outlier_mask = sub_flat.remove_outliers(
                sigma_lower=sigma_clip_lower,
                sigma_upper=sigma_clip_upper,
                return_mask=True,
            )
            full_outlier_mask = np.zeros(len(flat), dtype=bool)
            full_outlier_mask[non_transit_indices] = outlier_mask
            flat = flat[~full_outlier_mask]
        else:
            flat = flat.remove_outliers(
                sigma_lower=sigma_clip_lower,
                sigma_upper=sigma_clip_upper,
            )
    except Exception as exc:
        log.warning("Final outlier removal failed: %s", exc)

    if len(flat) == 0:
        raise PreprocessingError("Light curve is empty after preprocessing.")

    log.info("Preprocessing complete: %d cadences remain", len(flat))
    return flat
