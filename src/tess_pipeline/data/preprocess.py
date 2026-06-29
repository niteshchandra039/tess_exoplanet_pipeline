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
    sigma_clip_lower: float = 20.0,
    sigma_clip_upper: float = 5.0,
    flatten_window_length: int = 401,
    flatten_polyorder: int = 3,
    flatten_break_tolerance: int = 5,
) -> Any:
    """
    Preprocess a ``lightkurve.LightCurveCollection`` into a single
    clean, stitched, and lightly flattened light curve.

    Parameters
    ----------
    lc_collection : lightkurve.LightCurveCollection
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

    # ── Per-sector: remove NaNs and clip outliers ─────────────────────────────
    cleaned = []
    for lc in lc_collection:
        lc = lc.remove_nans()
        lc = lc.remove_outliers(
            sigma_lower=sigma_clip_lower,
            sigma_upper=sigma_clip_upper,
        )
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

    # ── Initial flatten (first-pass detrend) ─────────────────────────────────
    # NOTE: Do NOT over-detrend here; the GP noise model in bayesian.py
    # will handle residual correlated stellar variability.
    try:
        flat, trend = stitched.flatten(
            window_length=flatten_window_length,
            polyorder=flatten_polyorder,
            break_tolerance=flatten_break_tolerance,
            return_trend=True,
        )
    except Exception as exc:
        log.warning("Flattening failed (%s); using unflattened light curve", exc)
        flat = stitched

    # Final NaN pass and outlier removal after flattening
    flat = flat.remove_nans()
    try:
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
