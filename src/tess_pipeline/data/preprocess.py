"""
data/preprocess.py : Light curve preprocessing wrappers around lightkurve.

Steps:
  1. Remove NaNs
  2. Quality bit masking (applied per sector by lightkurve)
  3. Outlier sigma clipping (Per Sector)
  4. Initial flattening / detrending (Per Sector)
  5. Stitch multi-sector collection
  6. Generate and save diagnostic plots

Returns a single clean lightkurve.LightCurve object normalised to unit flux.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from tess_pipeline.exceptions import PreprocessingError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def preprocess(
    lc_collection: Any,
    *,
    period: float | None = None,
    epoch: float | None = None,
    transit_mask_width: float | None = None,
    sigma_clip_lower: float = 20.0,
    sigma_clip_upper: float = 5.0,
    flatten_window_length: int = 401,
    flatten_polyorder: int = 3,
    flatten_break_tolerance: int = 5,
    visualize: bool = False,
    save_plots: bool = True,
    output_dir: str = "preprocessing_plots",
) -> Any:
    """
    Preprocess a ``lightkurve.LightCurveCollection`` into a single
    clean, stitched, and flattened light curve.

    Parameters
    ----------
    lc_collection : Any
        A lightkurve.LightCurveCollection object containing the unstitched, raw observational data segments to be processed.
    period : float | None
        A float defining the orbital period of the target exoplanet in days. This is used to calculate the orbital 
        phase and generate a mask over the transits.
    epoch : float | None
        A float defining the transit mid-time reference point. If this is missing but a period is provided, the 
        script will attempt to guess the epoch using the timestamp of the lowest flux value in the dataset.
    transit_mask_width : float | None
        A float representing the total time window (in days) centered on the epoch that will be masked out. 
        If None and a period is provided, this scales automatically as 0.12 * P^(1/3). 
        If None and no period is provided, it falls back to 0.15 days.
    sigma_clip_lower : float
        A float determining the statistical threshold (in standard deviations) for rejecting negative flux anomalies. 
    sigma_clip_upper : float
        A float determining the statistical threshold for rejecting positive flux anomalies.
    flatten_window_length : int
        An integer specifying the number of cadences included in the moving window of the Savitzky-Golay filter. 
    flatten_polyorder : int
        An integer specifying the degree of the polynomial used within the Savitzky-Golay window.
    flatten_break_tolerance : int
        An integer defining the threshold for data gaps.
    visualize : bool
        A boolean flag. When set to True, the script will display the plots directly in the notebook or window.
    save_plots : bool
        A boolean flag. When set to True, the script saves all sector and final plots to the output directory.
    output_dir : str
        The folder path where diagnostic plots will be saved. Defaults to 'preprocessing_plots'.

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

    # Create the output directory if plotting is enabled
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        log.info("Saving diagnostic plots to directory: %s", output_dir)

    # Calculate dynamic mask width if not provided
    if transit_mask_width is None:
        if period is not None:
            transit_mask_width = 0.12 * (period ** (1.0 / 3.0))
            log.info("Dynamic mask width calculated: %.3f days for P = %.2f d", transit_mask_width, period)
        else:
            transit_mask_width = 0.15
            log.info("No period or mask width provided; falling back to default 0.15 days.")

    # Ensure odd window length for the Savitzky-Golay filter
    if flatten_window_length % 2 == 0:
        flatten_window_length += 1
        log.debug("Adjusted flatten_window_length to %d (must be odd)", flatten_window_length)

    cleaned_and_flat = []
    trends_list = []
    
    # Step 1 & 2: Per-sector cleaning, masking, and flattening
    for i, lc in enumerate(lc_collection):
        lc = lc.remove_nans()
        sector_name = getattr(lc, "sector", f"Index_{i}")

        # Create transit mask for clipping and flattening
        transit_mask = np.zeros(len(lc), dtype=bool)
        if period is not None:
            t0 = epoch
            if t0 is None:
                t0 = float(lc.time.value[np.argmin(lc.flux.value)])
            if t0 > 2400000 and np.median(lc.time.value) < 100000:
                t0 -= 2457000.0  
            phase = ((lc.time.value - t0) / period) % 1.0
            phase[phase > 0.5] -= 1.0
            transit_mask = np.abs(phase) < (transit_mask_width / period)
            
        # Initial Outlier Clipping
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
                transit_mask = transit_mask[~full_outlier_mask]
            else:
                lc = lc.remove_outliers(
                    sigma_lower=sigma_clip_lower,
                    sigma_upper=sigma_clip_upper,
                )
        except Exception as exc:
            log.warning("Outlier removal failed on sector: %s", exc)

        # Retain a copy of the pre-flattened light curve for plotting
        lc_pre_flat = lc.copy()

        # Flattening (Per-Sector Detrending)
        try:
            flat, trend = lc.flatten(
                window_length=flatten_window_length,
                polyorder=flatten_polyorder,
                break_tolerance=flatten_break_tolerance,
                mask=transit_mask if np.any(transit_mask) else None,
                return_trend=True,
            )
            lc = flat
            trends_list.append(trend)
        except Exception as exc:
            log.warning("Flattening failed (%s) on sector %s; using unflattened light curve", exc, sector_name)
            dummy_trend = lc.copy()
            dummy_trend.flux = np.ones_like(lc.flux.value)
            trends_list.append(dummy_trend)
            trend = dummy_trend

        # Final NaN pass and outlier removal after flattening
        lc = lc.remove_nans()
        
        final_mask = np.zeros(len(lc), dtype=bool)
        if period is not None:
            phase = ((lc.time.value - t0) / period) % 1.0
            phase[phase > 0.5] -= 1.0
            final_mask = np.abs(phase) < (transit_mask_width / period)

        try:
            if np.any(final_mask):
                non_transit_indices = np.where(~final_mask)[0]
                sub_flat = lc[non_transit_indices]
                _, outlier_mask = sub_flat.remove_outliers(
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
            log.warning("Final outlier removal failed on sector %s: %s", sector_name, exc)

        cleaned_and_flat.append(lc)

        # Plot Sector Results
        if visualize or save_plots:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            # Top Panel: Raw + Trend + Mask
            ax1.plot(lc_pre_flat.time.value, lc_pre_flat.flux.value, "k.", alpha=0.4, label="Pre-trend Data")
            if np.any(transit_mask):
                masked_lc = lc_pre_flat[transit_mask]
                ax1.plot(masked_lc.time.value, masked_lc.flux.value, "r.", alpha=0.8, label="Transit Mask")
            ax1.plot(trend.time.value, trend.flux.value, "C0-", lw=2, label="Savitzky-Golay Trend")
            
            ax1.set_ylabel("Flux")
            ax1.set_title(f"Sector {sector_name} Detrending")
            ax1.legend(loc="upper right")
            
            # Bottom Panel: Flattened Data
            ax2.plot(lc.time.value, lc.flux.value, "C1.", alpha=0.8, label="Flattened Data")
            ax2.axhline(1.0, color="k", ls="--", lw=1)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Normalized Flux")
            ax2.legend(loc="upper right")
            
            plt.tight_layout()
            
            if save_plots:
                sector_plot_path = os.path.join(output_dir, f"sector_{sector_name}_processing.png")
                plt.savefig(sector_plot_path, dpi=300, bbox_inches="tight")
                
            if visualize:
                plt.show()
            else:
                plt.close(fig)

    if not cleaned_and_flat:
        raise PreprocessingError("All sectors removed after NaN/outlier cleaning.")

    # Step 3: Stitch the independently flattened sectors together
    collection = lk.LightCurveCollection(cleaned_and_flat)
    stitched = collection.stitch()
    log.debug("Stitched %d sectors -> %d cadences", len(cleaned_and_flat), len(stitched))

    if len(stitched) == 0:
        raise PreprocessingError("Light curve is empty after preprocessing.")

    log.info("Preprocessing complete: %d cadences remain", len(stitched))

    # Step 4: Final 4:1 Visualization
    if visualize or save_plots:
        # Reconstruct a raw stitched curve for comparison
        raw_list = [raw_lc.remove_nans().normalize() for raw_lc in lc_collection]
        raw_stitched = lk.LightCurveCollection(raw_list).stitch()
        
        # Stitch the trendlines generated during the loop
        trend_stitched = lk.LightCurveCollection(trends_list).stitch()

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        # Top Panel: Raw vs Processed Overlay with Trend
        ax1.plot(raw_stitched.time.value, raw_stitched.flux.value, "k.", alpha=0.3, markersize=3, label="Raw (Normalized)")
        ax1.plot(trend_stitched.time.value, trend_stitched.flux.value, "C0-", lw=1.5, label="Savitzky-Golay Trend", zorder=5)
        ax1.plot(stitched.time.value, stitched.flux.value, "C1.", alpha=0.8, markersize=3, label="Processed (Flattened)")
        
        ax1.set_ylabel("Normalized Flux")
        ax1.legend(loc="upper right")
        ax1.set_title("Light Curve Preprocessing: Final Stitched Overview")
        
        # Hide top x-axis labels to avoid clutter
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Bottom Panel: Difference / Residual
        processed_interp = np.interp(raw_stitched.time.value, stitched.time.value, stitched.flux.value)
        residual = raw_stitched.flux.value - processed_interp
        
        ax2.plot(raw_stitched.time.value, residual, "k.", alpha=0.3, markersize=3)
        ax2.axhline(0, color="C1", ls="--", lw=1.5)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Difference")

        
        if save_plots:
            final_plot_path = os.path.join(output_dir, "final_stitched_preprocessing.png")
            plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
            log.info("Saved final stitched visualization to %s", final_plot_path)
            
        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return stitched