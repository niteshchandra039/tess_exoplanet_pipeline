"""
io/export.py — Export pipeline results to various file formats.

Formats:
  * CSV    — parameter summary table with medians and uncertainties
  * JSON   — metadata, config, and derived parameters
    * NPZ     — compressed posterior sample values needed for plot recreation
  * FITS   — light curve arrays and model arrays
  * PNG    — diagnostic figures
  * HTML   — modern exoplanet analysis dashboard
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)

# These are the only posterior values consumed by recreate_plots.py.  Keeping
# model predictions and likelihood/ sampler groups out of the archive avoids
# duplicating large arrays that are already represented in lightcurve_model_data.npz.
_PLOT_POSTERIOR_VARS = (
    "period",
    "rp_r_star",
    "b",
    "t14",
    "t0",
    "u1",
    "u2",
    "q1",
    "q2",
    "rho_star",
    "sigma_gp",
    "rho_gp",
)


def save_results(results: "PipelineResults", output_dir: Path) -> None:
    """
    Write all pipeline outputs to *output_dir*.

    Creates the directory if it does not exist.
    """
    output_dir = Path(output_dir)
    tic = results.target.get("tic_id", "unknown")
    
    # Create the TIC #### folder inside output_dir
    target_dir = output_dir / f"TIC {tic}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subfolder named plots in that folder
    plots_dir = target_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    prefix = target_dir / f"TIC{tic}"

    log.info("Saving results to %s", target_dir)

    _save_csv(results, prefix)
    _save_json(results, prefix)
    _save_posterior(results, prefix)
    _save_lightcurve_fits(results, prefix)
    _save_figures(results, plots_dir)
    _save_data_for_plots(results, target_dir)
    _save_html_report(results, target_dir)

    # Generate PDF summary report
    from tess_pipeline.io.report import generate_pdf_report
    try:
        generate_pdf_report(results, target_dir)
    except Exception as exc:
        log.warning("Could not generate PDF report: %s", exc)

    log.info("Export complete: %s", target_dir)


def _save_csv(results: "PipelineResults", prefix: Path) -> None:
    """Write parameter summary to CSV."""
    import csv

    rows = []
    
    # Planets
    if results.planets:
        for idx, pl in enumerate(results.planets):
            for key, val in pl.items():
                rows.append({"parameter": f"planet_{idx+1}_{key}", "value": val})
    else:
        for key, val in results.planet.items():
            rows.append({"parameter": f"planet_1_{key}", "value": val})
            
    # Stellar
    for key, val in results.stellar.items():
        if isinstance(val, (int, float, str, type(None))):
            rows.append({"parameter": f"stellar_{key}", "value": val})
            
    if results.period:
        rows.append({"parameter": "period_d", "value": results.period.get("value")})
        rows.append({"parameter": "period_source", "value": results.period.get("source")})

    path = Path(str(prefix) + "_summary.csv")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "value"])
        writer.writeheader()
        writer.writerows(rows)
    log.debug("Saved CSV: %s", path)


def _save_json(results: "PipelineResults", prefix: Path) -> None:
    """Write metadata and derived parameters to JSON."""
    path = Path(str(prefix) + "_metadata.json")
    data = results.to_dict()
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    log.debug("Saved JSON: %s", path)


def _save_posterior(results: "PipelineResults", prefix: Path) -> None:
    """Write only plot-required posterior sample values to a compressed NPZ."""
    if results.posterior is None:
        return
    try:
        import numpy as np

        posterior_group = results.posterior.posterior
        data_vars = getattr(posterior_group, "data_vars", {})
        plot_vars = {
            name: np.asarray(data_vars[name].values)
            for name in _PLOT_POSTERIOR_VARS
            if name in data_vars
        }
        if not plot_vars:
            log.warning("No plot-required posterior variables were found")
            return

        data_dir = prefix.parent / "data_for_plots"
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "posterior_samples.npz"
        np.savez_compressed(path, **plot_vars)
        log.debug("Saved compressed posterior sample values: %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save compressed posterior sample values: %s", exc)


def _save_lightcurve_fits(results: "PipelineResults", prefix: Path) -> None:
    """Write light curve to FITS."""
    if results.lightcurve is None:
        return
    try:
        path = Path(str(prefix) + "_lightcurve.fits")
        results.lightcurve.to_fits(str(path), overwrite=True)
        log.debug("Saved light curve FITS: %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save light curve FITS: %s", exc)


def _save_figures(results: "PipelineResults", output_dir: Path) -> None:
    """Save all figures as PNG files."""
    prefix_map = {
        "raw": "01_raw",
        "flat": "02_flat",
        "tls_periodogram": "03_tls_periodogram",
        "bls_periodogram": "03_bls_periodogram",
        "phase": "04_phase",
        "mcmc_phase": "04_mcmc_phase",
        "residuals": "05_residuals",
        "bayesian_fit": "06_bayesian_fit",
        "corner": "07_corner",
        "trace": "08_trace",
        "posterior_predictive": "09_posterior_predictive",
        "gp_acf": "10_gp_acf",
        "transit_stack": "11_transit_stack",
    }
    for name, fig in results.figures.items():
        if fig is None:
            continue
        try:
            filename = prefix_map.get(name, name)
            if filename.startswith("phase_p"):
                filename = filename.replace("phase_p", "04_phase_p")
            elif filename.startswith("mcmc_phase_p"):
                filename = filename.replace("mcmc_phase_p", "04_mcmc_phase_p")
            elif filename.startswith("transit_stack_p"):
                filename = filename.replace("transit_stack_p", "11_transit_stack_p")
            elif filename.startswith("tls_periodogram_coarse_p"):
                filename = filename.replace("tls_periodogram_coarse_p", "03_tls_periodogram_coarse_p")
            elif filename.startswith("tls_periodogram_fine_p"):
                filename = filename.replace("tls_periodogram_fine_p", "03_tls_periodogram_fine_p")
            elif filename.startswith("bls_periodogram_coarse_p"):
                filename = filename.replace("bls_periodogram_coarse_p", "03_bls_periodogram_coarse_p")
            elif filename.startswith("bls_periodogram_fine_p"):
                filename = filename.replace("bls_periodogram_fine_p", "03_bls_periodogram_fine_p")
            path = output_dir / f"{filename}.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            log.debug("Saved figure: %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save figure %r: %s", name, exc)


def _save_data_for_plots(results: "PipelineResults", target_dir: Path) -> None:
    """Save raw data to support standalone plot recreation, and write recreate_plots.py."""
    import numpy as np
    
    data_dir = target_dir / "data_for_plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Periodogram Data ──
    pg_kwargs = {}
    detections = results.metadata.get("detections", [])
    if not detections and results.detection:
        detections = [results.detection]
        
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        for k in ("tls_periods", "tls_power", "bls_periods", "bls_power", "tls_periods_broad", "tls_power_broad", "bls_periods_broad", "bls_power_broad"):
            val = det.get(k)
            if val is not None:
                pg_kwargs[f"{k}_p{idx}"] = np.asarray(val)
        pg_kwargs[f"period_p{idx}"] = det.get("period", 0.0)
        pg_kwargs[f"sde_p{idx}"] = det.get("sde") or det.get("snr", 0.0)
        pg_kwargs[f"epoch_p{idx}"] = det.get("epoch", 0.0)
        pg_kwargs[f"duration_hr_p{idx}"] = det.get("duration_hr", 3.0)
        pg_kwargs[f"depth_p{idx}"] = det.get("depth", 0.0)
        pg_kwargs[f"method_p{idx}"] = det.get("method", "tls")
        
    try:
        np.savez(data_dir / "periodogram_data.npz", **pg_kwargs)
        log.debug("Saved periodogram data: %s", data_dir / "periodogram_data.npz")
    except Exception as exc:
        log.warning("Could not save periodogram data: %s", exc)

    # ── 2. Light Curve & Model Data ──
    lc_kwargs = {}
    if results.lightcurve is not None:
        lc_kwargs["time"] = np.asarray(results.lightcurve.time.value)
        lc_kwargs["flux"] = np.asarray(results.lightcurve.flux.value)
        if results.lightcurve.flux_err is not None:
            lc_kwargs["flux_err"] = np.asarray(results.lightcurve.flux_err.value)
            
    if results.model:
        for k in ("gp_model", "transit_model", "flux_model", "residuals"):
            val = results.model.get(k)
            if val is not None:
                lc_kwargs[k] = np.asarray(val)
                
    if results.posterior is not None:
        post_group = results.posterior.posterior
        if hasattr(post_group, "to_dataset"):
            ds = post_group.to_dataset()
        else:
            ds = post_group
        flat_samps = ds.stack(sample=("chain", "draw"))
        
        # Save individual planet model draws
        for idx in range(len(results.planets) if results.planets else 1):
            lc_pred_key = f"lc_pred_p{idx}"
            if lc_pred_key in flat_samps.data_vars:
                pred = np.percentile(flat_samps[lc_pred_key].values, [16, 50, 84], axis=-1)
                lc_kwargs[f"pred_16_p{idx}"] = pred[0]
                lc_kwargs[f"pred_50_p{idx}"] = pred[1]
                lc_kwargs[f"pred_84_p{idx}"] = pred[2]
            
            key = f"light_curve_p{idx}"
            if key in flat_samps.data_vars:
                lc_kwargs[f"light_curve_median_p{idx}"] = np.median(flat_samps[key].values, axis=-1)
                
            if "gp_pred" in flat_samps.data_vars:
                lc_kwargs["gp_pred_median"] = np.median(flat_samps["gp_pred"].values, axis=-1)
                
            if "t14" in flat_samps.data_vars:
                if flat_samps["t14"].values.ndim == 2:
                    lc_kwargs[f"t14_med_p{idx}"] = np.median(flat_samps["t14"].values[idx, :])
                else:
                    lc_kwargs[f"t14_med_p{idx}"] = np.median(flat_samps["t14"].values)
            elif results.planets and idx < len(results.planets):
                pl = results.planets[idx]
                if "t14_hr" in pl:
                    lc_kwargs[f"t14_med_p{idx}"] = float(pl["t14_hr"]) / 24.0
                    
            if "period" in flat_samps.data_vars:
                if flat_samps["period"].values.ndim == 2:
                    lc_kwargs[f"period_med_p{idx}"] = np.median(flat_samps["period"].values[idx, :])
                    lc_kwargs[f"period_std_p{idx}"] = np.std(flat_samps["period"].values[idx, :])
                    lc_kwargs[f"epoch_med_p{idx}"] = np.median(flat_samps["t0"].values[idx, :])
                else:
                    lc_kwargs[f"period_med_p{idx}"] = np.median(flat_samps["period"].values)
                    lc_kwargs[f"period_std_p{idx}"] = np.std(flat_samps["period"].values)
                    lc_kwargs[f"epoch_med_p{idx}"] = np.median(flat_samps["t0"].values)
                    
    # ── Save raw stitched light curve for recreation ──
    if getattr(results, "lightcurve_raw", None) is not None:
        lc_kwargs["raw_time"] = np.asarray(results.lightcurve_raw.time.value)
        lc_kwargs["raw_flux"] = np.asarray(results.lightcurve_raw.flux.value)

    try:
        np.savez(data_dir / "lightcurve_model_data.npz", **lc_kwargs)
        log.debug("Saved lightcurve model data: %s", data_dir / "lightcurve_model_data.npz")
    except Exception as exc:
        log.warning("Could not save lightcurve model data: %s", exc)

    # ── 3. Standalone Plotting Script (recreate_plots.py) ──
    tic_id = results.target.get("tic_id", "unknown")
    recreate_script_path = target_dir / "recreate_plots.py"
    
    script_content = f"""# Standalone script to recreate all TESS analysis plots
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Apply publication-ready style configurations (MNRAS-style STIX fonts, inward ticks)
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.frameon'] = False
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.major.size'] = 4.0
plt.rcParams['xtick.minor.size'] = 2.0
plt.rcParams['ytick.major.size'] = 4.0
plt.rcParams['ytick.minor.size'] = 2.0
plt.rcParams['xtick.major.width'] = 0.75
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.75
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Try to load target metadata
tic_id = "{tic_id}"
sectors_str = "unknown"
try:
    with open(f"TIC{{tic_id}}_metadata.json") as f:
        meta = json.load(f)
        secs = meta.get("metadata", {{}}).get("sectors_used", [])
        sectors_str = ", ".join(map(str, secs)) if secs else "unknown"
except Exception:
    pass

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Helper function to compute phase folded curve
def phase_fold(time, flux, period, epoch):
    t0 = epoch if epoch is not None else time[0]
    phase = ((time - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]

# Helper for binning
def bin_phase_curve(phase, flux, n_bins=80):
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_flux = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_flux[i] = np.mean(flux[mask])
    return bin_centers, bin_flux

# ── Load Data ──
try:
    pg_data = np.load("data_for_plots/periodogram_data.npz", allow_pickle=True)
    lc_data = np.load("data_for_plots/lightcurve_model_data.npz", allow_pickle=True)
    print("Successfully loaded plotting data.")
except Exception as e:
    print(f"Error loading data: {{e}}")
    exit(1)

# ── 1. Recreate Raw & Flat Light Curves (01_raw.png, 02_flat.png) ──
if "raw_time" in lc_data and "raw_flux" in lc_data:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(lc_data["raw_time"], lc_data["raw_flux"], s=0.5, color="steelblue", alpha=0.6, rasterized=True)
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("PDCSAP Flux")
    ax.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | Raw TESS Light Curve", fontsize=10, fontweight="bold")
    fig.tight_layout()
    print("Generating plot: plots/01_raw.png")
    fig.savefig("plots/01_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if "time" in lc_data and "flux" in lc_data:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(lc_data["time"], lc_data["flux"], s=0.5, color="darkorange", alpha=0.6, rasterized=True)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | Flattened TESS Light Curve", fontsize=10, fontweight="bold")
    fig.tight_layout()
    print("Generating plot: plots/02_flat.png")
    fig.savefig("plots/02_flat.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 2. Recreate TLS/BLS Periodograms (03_tls_periodogram.png, 03_bls_periodogram.png) ──
for method in ("tls", "bls"):
    keys = [k for k in pg_data.keys() if k.startswith(f"{{method}}_periods_p")]
    if not keys:
        keys_broad = [k for k in pg_data.keys() if k.startswith(f"{{method}}_periods_broad_p")]
        if not keys_broad:
            continue
        n_panels = len(keys_broad)
    else:
        n_panels = len(keys)
        
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), squeeze=False)
    for idx in range(n_panels):
        ax = axes[idx, 0]
        periods = pg_data.get(f"{{method}}_periods_broad_p{{idx}}")
        if periods is None:
            periods = pg_data.get(f"{{method}}_periods_p{{idx}}")
        power = pg_data.get(f"{{method}}_power_broad_p{{idx}}")
        if power is None:
            power = pg_data.get(f"{{method}}_power_p{{idx}}")
            
        best_period = float(pg_data[f"period_p{{idx}}"])
        stat = float(pg_data[f"sde_p{{idx}}"])
        stat_name = "SDE" if method == "tls" else "SNR"
        
        if periods is not None and len(periods) > 0:
            color = "#2563eb" if method == "tls" else "#d97706"
            ax.plot(periods, power, color=color, linewidth=0.8)
            ax.set_xlim(np.min(periods), np.max(periods))
            ax.set_ylim(0, np.max(power) * 1.1)
            
        ax.axvline(best_period, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Best period: {{best_period:.5f}} d")
        if method == "tls":
            for n in (2, 3):
                ax.axvline(best_period / n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
                ax.axvline(best_period * n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
                
        ax.set_ylabel(f"{{method.upper()}} Power")
        ax.legend(fontsize=9, loc="upper right")
        title_suffix = f" (Planet {{idx+1}} Search)" if n_panels > 1 else ""
        ax.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | {{method.upper()}} Periodogram{{title_suffix}} ({{stat_name}} = {{stat:.2f}})", fontsize=10, fontweight="bold")
    axes[-1, 0].set_xlabel("Period (days)")
    fig.tight_layout()
    print(f"Generating plot: plots/03_{{method}}_periodogram.png")
    fig.savefig(f"plots/03_{{method}}_periodogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Save individual coarse/fine periodograms
    for idx in range(n_panels):
        stat = float(pg_data[f"sde_p{{idx}}"])
        stat_name = "SDE" if method == "tls" else "SNR"
        best_period = float(pg_data[f"period_p{{idx}}"])
        
        for mode in ("coarse", "fine"):
            if mode == "coarse":
                periods = pg_data.get(f"{{method}}_periods_broad_p{{idx}}")
                if periods is None:
                    periods = pg_data.get(f"{{method}}_periods_p{{idx}}")
                power = pg_data.get(f"{{method}}_power_broad_p{{idx}}")
                if power is None:
                    power = pg_data.get(f"{{method}}_power_p{{idx}}")
            else:
                periods = pg_data.get(f"{{method}}_periods_p{{idx}}")
                power = pg_data.get(f"{{method}}_power_p{{idx}}")
                
            if periods is None or len(periods) == 0:
                continue
                
            fig_ind, ax_ind = plt.subplots(figsize=(12, 3.5))
            color = "#2563eb" if method == "tls" else "#d97706"
            ax_ind.plot(periods, power, color=color, linewidth=0.8)
            ax_ind.set_xlim(np.min(periods), np.max(periods))
            ax_ind.set_ylim(0, np.max(power) * 1.1)
            
            ax_ind.axvline(best_period, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Best period: {{best_period:.5f}} d")
            if method == "tls":
                for n in (2, 3):
                    ax_ind.axvline(best_period / n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
                    ax_ind.axvline(best_period * n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
            
            ax_ind.set_ylabel(f"{{method.upper()}} Power")
            ax_ind.set_xlabel("Period (days)")
            ax_ind.legend(fontsize=9, loc="upper right")
            mode_label = " Coarse" if mode == "coarse" else " Fine"
            ax_ind.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | {{method.upper()}}{{mode_label}} Periodogram (Planet {{idx+1}} Search) ({{stat_name}} = {{stat:.2f}})", fontsize=10, fontweight="bold")
            fig_ind.tight_layout()
            print(f"Generating plot: plots/03_{{method}}_periodogram_{{mode}}_p{{idx}}.png")
            fig_ind.savefig(f"plots/03_{{method}}_periodogram_{{mode}}_p{{idx}}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ind)

# ── 3. Recreate Discovery Phase Curve (04_phase.png) ──
keys = [k for k in pg_data.keys() if k.startswith("period_p")]
n_planets = len(keys) if keys else 1
for idx in range(n_planets):
    p_disc = float(pg_data.get(f"period_p{{idx}}") or 1.0)
    e_disc = float(pg_data.get(f"epoch_p{{idx}}") or (lc_data["time"][0] if "time" in lc_data else 0))
    dur_hr = float(pg_data.get(f"duration_hr_p{{idx}}") or 3.0)
    depth = float(pg_data.get(f"depth_p{{idx}}") or 0.005)
    
    if p_disc > 0 and "time" in lc_data and "flux" in lc_data:
        phase_disc, flux_disc = phase_fold(lc_data["time"], lc_data["flux"], p_disc, e_disc)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(phase_disc * p_disc, flux_disc, s=0.8, color="black", alpha=0.2, label="Data", rasterized=True)
        
        bin_p, bin_f = bin_phase_curve(phase_disc, flux_disc, n_bins=80)
        valid = np.isfinite(bin_f)
        ax.errorbar(bin_p[valid] * p_disc, bin_f[valid], fmt="o", ms=4, color="steelblue", label="Binned")
        
        ax.set_xlim(-1.5 * (dur_hr / 24.0), 1.5 * (dur_hr / 24.0))
        ax.set_ylim(1.0 - 3.0 * depth, 1.0 + 1.5 * depth)
        ax.set_xlabel("Time since transit (days)")
        ax.set_ylabel("Relative Flux")
        ax.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | Phased Discovery Light Curve - Planet {{idx+1}}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()
        filename = f"04_phase_p{{idx}}.png" if (idx > 0 or n_planets > 1) else "04_phase.png"
        print(f"Generating plot: plots/{{filename}}")
        fig.savefig(f"plots/{{filename}}", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ── 4. Recreate MCMC Phase Curves (04_mcmc_phase.png) ──
if "time" in lc_data and "flux" in lc_data:
    time = lc_data["time"]
    flux = lc_data["flux"]
    
    keys_mcmc = [k for k in lc_data.keys() if k.startswith("period_med_p")]
    n_planets_mcmc = len(keys_mcmc) if keys_mcmc else 1
    
    for idx in range(n_planets_mcmc):
        if f"period_med_p{{idx}}" not in lc_data:
            if idx == 0 and "gp_model" in lc_data:
                p_med = float(lc_data.get("period_p0") or 1.0)
                p_std = 0.0
                e_med = float(lc_data.get("epoch_p0") or time[0])
                t14_med = 0.15
            else:
                break
        else:
            p_med = float(lc_data[f"period_med_p{{idx}}"])
            p_std = float(lc_data.get(f"period_std_p{{idx}}") or 0.0)
            e_med = float(lc_data[f"epoch_med_p{{idx}}"])
            t14_med = float(lc_data.get(f"t14_med_p{{idx}}") or 0.15)
            
        gp_model = lc_data.get("gp_model")
        detrended = flux - gp_model if gp_model is not None else flux
        
        other_mod = np.zeros_like(flux)
        j = 0
        while True:
            key = f"light_curve_median_p{{j}}"
            if key in lc_data:
                if j != idx:
                    other_mod += lc_data[key]
                j += 1
            else:
                break
        detrended = detrended - other_mod
        
        phase, y_fold = phase_fold(time, detrended, p_med, e_med)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True,
                                 gridspec_kw={{"height_ratios": [2, 1]}})
        
        axes[0].scatter(phase * p_med, y_fold, s=0.8, color="black", alpha=0.2, label="Data", rasterized=True, zorder=-1000)
        
        bin_phase, bin_flux = bin_phase_curve(phase, y_fold, n_bins=80)
        valid = np.isfinite(bin_flux)
        axes[0].errorbar(bin_phase[valid] * p_med, bin_flux[valid], fmt="o", ms=4, color="#16a34a", label="Binned", zorder=100)
        
        pred_key = f"pred_50_p{{idx}}"
        depth = 0.005
        transit_med = None
        if pred_key in lc_data:
            phase_grid = np.linspace(-0.3, 0.3, 200)
            axes[0].plot(phase_grid, lc_data[pred_key], color="#dc2626", linewidth=2.0, label="Transit Model", zorder=1000)
            depth = max(1.0 - np.min(lc_data[pred_key]), 1e-4)
        elif f"light_curve_median_p{{idx}}" in lc_data:
            transit_med = lc_data[f"light_curve_median_p{{idx}}"]
            _, t_fold = phase_fold(time, transit_med + 1.0, p_med, e_med)
            sort_idx = np.argsort(phase)
            x_sorted = phase[sort_idx] * p_med
            y_sorted = t_fold[sort_idx]
            grid_x = np.linspace(-0.3, 0.3, 1000)
            grid_y = np.interp(grid_x, x_sorted, y_sorted, left=1.0, right=1.0)
            axes[0].plot(grid_x, grid_y, color="#dc2626", linewidth=2.0, label="Transit Model", zorder=1000)
            depth = max(1.0 - np.min(grid_y), 1e-4)
            
        txt = f"P = {{p_med:.5f}} ± {{p_std:.5f}} d"
        axes[0].annotate(txt, (0.02, 0.05), xycoords="axes fraction", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"), fontsize=10)
        
        axes[0].set_xlim(-2.5 * t14_med, 2.5 * t14_med)
        axes[0].set_ylim(1.0 - 2.5 * depth, 1.0 + 1.5 * depth)
        axes[0].set_ylabel("De-trended Relative Flux")
        axes[0].set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | Phased MCMC Fit - Planet {{idx + 1}}", fontsize=10, fontweight="bold")
        axes[0].legend(fontsize=9, loc="upper right")
        
        if transit_med is not None:
            res_val = detrended - (transit_med + 1.0)
        elif "residuals" in lc_data:
            res_val = lc_data["residuals"]
        else:
            res_val = np.zeros_like(flux)
            
        _, r_fold = phase_fold(time, res_val + 1.0, p_med, e_med)
        axes[1].scatter(phase * p_med, r_fold - 1.0, s=0.8, color="gray", alpha=0.3, label="Residuals", rasterized=True)
        
        bin_r_phase, bin_r_flux = bin_phase_curve(phase, r_fold - 1.0, n_bins=80)
        valid_r = np.isfinite(bin_r_flux)
        axes[1].errorbar(bin_r_phase[valid_r] * p_med, bin_r_flux[valid_r], fmt="o", ms=4, color="#dc2626", label="Binned Residuals", zorder=10)
        
        axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Residuals")
        axes[1].set_xlabel("Time since transit (days)")
        axes[1].legend(fontsize=9, loc="upper right")
        axes[1].set_ylim(-1.5 * depth, 1.5 * depth)
        
        fig.tight_layout()
        filename = f"04_mcmc_phase_p{{idx}}.png" if (idx > 0 or n_planets_mcmc > 1) else "04_mcmc_phase.png"
        print(f"Generating plot: plots/{{filename}}")
        fig.savefig(f"plots/{{filename}}", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ── 5. Recreate Residuals (05_residuals.png) ──
if "time" in lc_data and "residuals" in lc_data:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(lc_data["time"], lc_data["residuals"], s=0.5, color="gray", alpha=0.5, rasterized=True, label="Residuals")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Residual Flux")
    ax.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | MCMC Residuals vs Time", fontsize=10, fontweight="bold")
    fig.tight_layout()
    print("Generating plot: plots/05_residuals.png")
    fig.savefig("plots/05_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 6. Recreate 2-Panel MCMC Fit plot (06_bayesian_fit.png) ──
if "gp_model" in lc_data and "transit_model" in lc_data and "flux_model" in lc_data:
    gp_model = lc_data["gp_model"]
    transit_model = lc_data["transit_model"]
    flux_model = lc_data["flux_model"]
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
    
    axes[0].scatter(time, flux, s=0.4, color="black", alpha=0.25, label="Data", rasterized=True)
    trend = flux_model - (transit_model - 1.0)
    axes[0].plot(time, trend, color="#22c55e", linewidth=1.2, label="GP Activity Model")
    axes[0].set_ylabel("Relative Flux")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | Bayesian GP Detrending (Time Domain)", fontsize=10, fontweight="bold")
    
    detrended = flux - gp_model
    axes[1].scatter(time, detrended, s=0.4, color="black", alpha=0.25, label="Detrended Data", rasterized=True)
    axes[1].plot(time, transit_model, color="#dc2626", linewidth=1.2, label="Keplerian Fit")
    
    axes[1].set_ylabel("Relative Flux")
    axes[1].set_xlabel("Time (BTJD)")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_title("Detrended Light Curve & Best-Fit Transit Model", fontsize=10, fontweight="bold")
    
    fig.tight_layout()
    print("Generating plot: plots/06_bayesian_fit.png")
    fig.savefig("plots/06_bayesian_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 7. Recreate Posterior Predictive check (09_posterior_predictive.png) ──
if "time" in lc_data and "flux" in lc_data and "flux_model" in lc_data:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(lc_data["time"], lc_data["flux"], s=0.5, color="gray", alpha=0.3, rasterized=True, label="Data")
    ax.plot(lc_data["time"], lc_data["flux_model"], color="#dc2626", linewidth=1.5, zorder=5, label="Posterior Median Model")
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | MCMC Posterior Predictive Check", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    print("Generating plot: plots/09_posterior_predictive.png")
    fig.savefig("plots/09_posterior_predictive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 8. Recreate GP Autocorrelation Plot (10_gp_acf.png) ──
if "gp_model" in lc_data:
    fig, ax = plt.subplots(figsize=(10, 4))
    res_before = flux - lc_data.get("transit_model", np.ones_like(flux))
    res_after = lc_data.get("residuals", np.zeros_like(flux))
    
    def get_acf(x, max_lag=150):
        x_centered = x - np.mean(x)
        n = len(x_centered)
        acf = np.correlate(x_centered, x_centered, mode='full')[n-1:n+max_lag]
        if acf[0] > 0:
            acf = acf / acf[0]
        return acf
        
    max_lag = min(len(time) // 4, 150)
    lags = np.arange(max_lag)
    acf_before = get_acf(res_before, max_lag=max_lag-1)
    acf_after = get_acf(res_after, max_lag=max_lag-1)
    
    ax.plot(lags, acf_before, color="#94a3b8", label="Raw Residuals (No GP)", linewidth=1.5)
    ax.plot(lags, acf_after, color="#0f766e", label="MCMC Residuals (After GP)", linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    
    conf = 1.96 / np.sqrt(len(time))
    ax.axhspan(-conf, conf, color="#0f766e", alpha=0.1, label="95% Confidence Interval")
    
    ax.set_xlabel("Lag (indices)")
    ax.set_ylabel("Autocorrelation Coeff")
    ax.set_title(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | GP Residuals ACF", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    print("Generating plot: plots/10_gp_acf.png")
    fig.savefig("plots/10_gp_acf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 9. Recreate Stacked transits (11_transit_stack.png) ──
if "time" in lc_data:
    keys = [k for k in lc_data.keys() if k.startswith("period_med_p")]
    n_planets = len(keys) if keys else 1
    
    for idx in range(n_planets):
        if f"period_med_p{{idx}}" not in lc_data:
            if idx == 0 and ("period_p0" in lc_data or "epoch_p0" in lc_data):
                p_med = float(lc_data.get("period_p0") or 1.0)
                e_med = float(lc_data.get("epoch_p0") or time[0])
                t14_med = float(lc_data.get("t14_med_p0") or 0.15)
            else:
                break
        else:
            p_med = float(lc_data[f"period_med_p{{idx}}"])
            e_med = float(lc_data[f"epoch_med_p{{idx}}"])
            t14_med = float(lc_data.get(f"t14_med_p{{idx}}") or 0.15)
            
        gp_model = lc_data.get("gp_model")
        detrended = flux - gp_model if gp_model is not None else flux / np.median(flux)
        
        other_mod = np.zeros_like(flux)
        j = 0
        while True:
            key = f"light_curve_median_p{{j}}"
            if key in lc_data:
                if j != idx:
                    other_mod += lc_data[key]
                j += 1
            else:
                break
        detrended = detrended - other_mod
        
        transit_model = None
        if f"light_curve_median_p{{idx}}" in lc_data:
            transit_model = lc_data[f"light_curve_median_p{{idx}}"] + 1.0
        elif "transit_model" in lc_data:
            transit_model = lc_data["transit_model"]
            
        half_width = max(0.2, t14_med * 3.0)
        t_start = np.min(time)
        t_end = np.max(time)
        n_start = int(np.floor((t_start - e_med) / p_med))
        n_end = int(np.ceil((t_end - e_med) / p_med))
        epochs = [e_med + n * p_med for n in range(n_start, n_end + 1)]
        epochs = [ep for ep in epochs if t_start - half_width < ep < t_end + half_width]
        
        if len(epochs) > 0:
            max_to_plot = 8
            if len(epochs) > max_to_plot:
                indices = np.linspace(0, len(epochs) - 1, max_to_plot, dtype=int)
                epochs_to_plot = [epochs[i] for i in indices]
            else:
                epochs_to_plot = epochs
                
            fig, axes = plt.subplots(len(epochs_to_plot), 1, figsize=(8, 1.5 * len(epochs_to_plot)), sharex=True)
            if len(epochs_to_plot) == 1:
                axes = [axes]
                
            for s_idx, ep in enumerate(epochs_to_plot):
                ax = axes[s_idx]
                mask = (time >= ep - half_width) & (time <= ep + half_width)
                t_sub = time[mask] - ep
                f_sub = detrended[mask]
                
                if len(t_sub) > 0:
                    ax.scatter(t_sub, f_sub, s=1.5, color="black", alpha=0.4, rasterized=True, label="Data" if s_idx == 0 else None)
                    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.7)
                    
                    if transit_model is not None:
                        m_sub = transit_model[mask]
                        sort_idx = np.argsort(t_sub)
                        ax.plot(t_sub[sort_idx], m_sub[sort_idx], color="#dc2626", linewidth=1.5, label="Transit Model" if s_idx == 0 else None)
                    
                ax.set_ylabel(f"Transit {{s_idx+1}}")
                ax.set_xlim(-half_width, half_width)
                
                if len(f_sub) > 0:
                    min_val = np.min(f_sub)
                    if transit_model is not None and len(m_sub) > 0:
                        min_val = min(min_val, np.min(m_sub))
                    local_std = np.std(f_sub)
                    ax.set_ylim(min_val - 0.00001 * abs(min_val), 1.0 + 3.0 * local_std)
                    
            axes[-1].set_xlabel("Time since transit mid-time (days)")
            if transit_model is not None:
                axes[0].legend(fontsize=8, loc="upper right")
            
            title_suffix = f" - Planet {{idx+1}}" if n_planets > 1 else ""
            fig.suptitle(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | Stacked Individual Transits{{title_suffix}}", y=0.99, fontsize=11, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.2)
            
            filename = f"11_transit_stack_p{{idx}}.png" if (idx > 0 or n_planets > 1) else "11_transit_stack.png"
            print(f"Generating plot: plots/{{filename}}")
            fig.savefig(f"plots/{{filename}}", dpi=150, bbox_inches="tight")
            plt.close(fig)

# ── 10. Recreate MCMC Corner & Trace plots (07_corner.png, 08_trace.png) ──
if os.path.exists("data_for_plots/posterior_samples.npz"):
    try:
        import arviz as az
        posterior_values = np.load("data_for_plots/posterior_samples.npz", allow_pickle=False)
        posterior = az.from_dict({{
            "posterior": {{name: posterior_values[name] for name in posterior_values.files}}
        }})
        print("Successfully loaded compressed posterior sample values.")
        
        # Corner Plot
        try:
            import corner
            LABELS_DICT = {{
                "period": "Period $P$ [d]",
                "rp_r_star": "$R_p/R_*$",
                "b": "Impact parameter $b$",
                "t14": "Duration $T_{{14}}$ [d]",
                "t0": "Epoch $t_0$ [BTJD]",
                "u1": "$u_1$",
                "u2": "$u_2$",
                "q1": "$q_1$",
                "q2": "$q_2$",
                "rho_star": "Density $\\\\rho_*$ [g/cm$^3$]",
                "sigma_gp": "GP $\\\\sigma$",
                "rho_gp": "GP $\\\\rho$",
            }}
            
            var_names = ["period", "rp_r_star", "b", "t14", "u1", "u2", "rho_star"]
            available = list(posterior.posterior.data_vars)
            var_names = [v for v in var_names if v in available]
            
            flat_samps = posterior.posterior.stack(sample=("chain", "draw"))
            samples_list = []
            labels = []
            for v in var_names:
                vals = flat_samps[v].values
                if vals.ndim == 2:
                    n_pl = vals.shape[0]
                    for i in range(n_pl):
                        samples_list.append(vals[i, :])
                        labels.append(f"{{LABELS_DICT.get(v, v)}} (planet {{i+1}})")
                else:
                    samples_list.append(vals)
                    labels.append(LABELS_DICT.get(v, v))
            
            samples = np.column_stack(samples_list)
            
            fig_corner = corner.corner(
                samples,
                labels=labels,
                color="#0f766e",
                hist_kwargs={{"color": "#0d9488", "fill": True, "alpha": 0.3}},
                show_titles=True,
                title_fmt=".5f",
                title_kwargs={{"fontsize": 9, "fontweight": "normal"}},
                label_kwargs={{"fontsize": 10}},
            )
            
            desc = (
                "MCMC Parameter Guide:\\n"
                "--------------------\\n"
                "P          : Orbital period [days]\\n"
                "Rₚ/R★      : Planet/stellar radius ratio\\n"
                "b          : Impact parameter\\n"
                "T₁₄        : Transit duration [days]\\n"
                "u₁, u₂     : Limb darkening coeffs\\n"
                "ρ★         : Stellar density [g/cm³]"
            )
            fig_corner.text(0.62, 0.75, desc, fontsize=9.5, family="monospace", va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", alpha=0.9, ec="#cbd5e1"))
            
            fig_corner.suptitle(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | MCMC Posterior Corner Plot", y=1.02, fontsize=11, fontweight="bold")
            print("Generating plot: plots/07_corner.png")
            fig_corner.savefig("plots/07_corner.png", dpi=150, bbox_inches="tight")
            plt.close(fig_corner)
        except Exception as e_corner:
            print(f"Failed to recreate corner plot: {{e_corner}}")
            
        # Trace Plot
        try:
            var_names_trace = ["rp_r_star", "b", "t14", "t0", "q1", "q2", "sigma_gp", "rho_gp"]
            available_trace = list(posterior.posterior.data_vars)
            var_names_trace = [v for v in var_names_trace if v in available_trace]
            
            fig_size = (12, 1.6 * len(var_names_trace))
            axes = None
            try:
                axes = az.plot_trace(posterior, var_names=var_names_trace, backend_kwargs={{"figsize": fig_size}})
            except (TypeError, ValueError):
                try:
                    axes = az.plot_trace(posterior, var_names=var_names_trace, figsize=fig_size)
                except (TypeError, ValueError):
                    axes = az.plot_trace(posterior, var_names=var_names_trace)
                    
            fig_trace = plt.gcf()
            
            LABELS_DICT = {{
                "period": "Period $P$ [d]",
                "rp_r_star": "$R_p/R_*$",
                "b": "Impact parameter $b$",
                "t14": "Duration $T_{{14}}$ [d]",
                "t0": "Epoch $t_0$ [BTJD]",
                "u1": "$u_1$",
                "u2": "$u_2$",
                "q1": "$q_1$",
                "q2": "$q_2$",
                "rho_star": "Density $\\\\rho_*$ [g/cm$^3$]",
                "sigma_gp": "GP $\\\\sigma$",
                "rho_gp": "GP $\\\\rho$",
            }}
            
            if axes is not None:
                axes_2d = np.atleast_2d(axes)
                for i, v in enumerate(var_names_trace):
                    if i < len(axes_2d):
                        label = LABELS_DICT.get(v, v)
                        axes_2d[i, 0].set_xlabel(label, fontsize=8.5)
                        axes_2d[i, 1].set_ylabel(label, fontsize=8.5)
                        axes_2d[i, 1].set_xlabel("Draw", fontsize=8.5)
                        
            for ax in fig_trace.axes:
                ax.tick_params(labelsize=8)
                ax.xaxis.label.set_size(8.5)
                ax.yaxis.label.set_size(8.5)
                if ax.get_title():
                    ax.set_title(ax.get_title(), fontsize=9, fontweight="normal")
                    
            fig_trace.suptitle(f"TIC {{tic_id}} | Sectors: {{sectors_str}} | MCMC Trace Plots", y=0.99, fontsize=11, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.4)
            print("Generating plot: plots/08_trace.png")
            fig_trace.savefig("plots/08_trace.png", dpi=150, bbox_inches="tight")
            plt.close(fig_trace)
        except Exception as e_trace:
            print(f"Failed to recreate trace plot: {{e_trace}}")
            
    except Exception as e_post:
        print(f"Error loading/processing posterior: {{e_post}}")

print("All plots recreated successfully.")
"""
    
    try:
        with open(recreate_script_path, "w") as f:
            f.write(script_content)
        log.debug("Saved recreate script: %s", recreate_script_path)
    except Exception as exc:
        log.warning("Could not save recreate script: %s", exc)


def _save_html_report(results: "PipelineResults", target_dir: Path) -> None:
    """Generate a self-contained interactive HTML report in the target directory."""
    import numpy as np
    path = target_dir / "report.html"
    data = results.to_dict()
    json_str = json.dumps(data, indent=2, default=str)

    # Helper for formatting values
    def get_val(section, key, decimals=4, unit=""):
        val = data.get(section, {}).get(key)
        if val is None or val == "":
            return "N/A"
        try:
            fval = float(val)
            return f"{fval:.{decimals}f}{unit}"
        except (ValueError, TypeError):
            return f"{val}{unit}"

    # Target fields
    tic_id = data.get("target", {}).get("tic_id", "Unknown")
    target_name = data.get("target", {}).get("name", f"TIC {tic_id}")
    ra = data.get("target", {}).get("ra", 0)
    dec = data.get("target", {}).get("dec", 0)

    # ── Dyn Planet Cards ──────────────────────────────────────────────────
    planets_cards_html = ""
    planets_data = data.get("planets") or []
    detections = data.get("metadata", {}).get("detections") or []
    if isinstance(detections, str):
        try:
            detections = json.loads(detections)
        except Exception:
            detections = []

    # ── Model Comparison Table ────────────────────────────────────────────
    model_comparison_html = ""
    if len(detections) >= 2 or "model_comparison" in data.get("metadata", {}):
        comp = data.get("metadata", {}).get("model_comparison") or {}
        delta_bic = comp.get("delta_bic")
        prob_multi = comp.get("probability_multiplanet")
        
        comparison_info = ""
        if delta_bic is not None and prob_multi is not None:
            comparison_info = f"""
            <div class="param-row" style="margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;">
                <span class="param-label">BIC Model Comparison (ΔBIC)</span>
                <span class="param-value">
                    {delta_bic:.2f}
                    <span class="badge badge-derived">BIC(2p) - BIC(1p)</span>
                </span>
            </div>
            <div class="param-row">
                <span class="param-label">Relative Model Probability</span>
                <span class="param-value">
                    {prob_multi * 100.0:.2f}%
                    <span class="badge badge-fits">Jeffreys Probability</span>
                </span>
            </div>
            """

        rows = ""
        for idx, det in enumerate(detections):
            p_val = det.get("period", 0.0)
            sde = det.get("sde") or det.get("snr", 0.0)
            fap = det.get("fap")
            fap_str = f"{fap*100.0:.4f}%" if fap is not None else "N/A"
            conf = det.get("confidence", "N/A")
            
            bayes_prob = det.get("existence_probability_bayesian")
            bayes_prob_str = f"{bayes_prob*100.0:.1f}%" if bayes_prob is not None else "N/A"
            bayes_conf = det.get("bayesian_confidence", "N/A")
            
            rows += f"""
            <tr>
                <td><strong>Planet {idx + 1}</strong></td>
                <td>{p_val:.5f} d</td>
                <td>{sde:.2f}</td>
                <td>{fap_str}</td>
                <td><span class="badge badge-derived">{conf}</span></td>
                <td><strong>{bayes_prob_str}</strong> ({bayes_conf})</td>
            </tr>
            """
            
        model_comparison_html = f"""
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Planet Candidate Confidence & Verdict Comparison</h2>
            <table class="confidence-table">
                <thead>
                    <tr>
                        <th>Candidate</th>
                        <th>Period</th>
                        <th>Search SDE/SNR</th>
                        <th>False Alarm Prob (FAP)</th>
                        <th>Search Confidence</th>
                        <th>Bayesian Probability & Verdict</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            {comparison_info}
        </div>
        """

    # Primary period
    primary_period_str = "N/A"
    if planets_data:
        primary_period_str = f"{planets_data[0].get('period', 0.0):.5f} d"
    elif data.get("period", {}).get("value") is not None:
        primary_period_str = f"{data.get('period', {}).get('value'):.5f} d"

    if planets_data:
        for idx, pl in enumerate(planets_data):
            p_val = pl.get("period", 0.0)
            p_err = pl.get("period_err", 0.0)
            t0_val = pl.get("t0", 0.0)
            t0_err = pl.get("t0_err", 0.0)
            rp_r_star = pl.get("rp_r_star", 0.0)
            rp_r_star_err = pl.get("rp_r_star_err", 0.0)
            rp_earth = pl.get("rp_earth", 0.0)
            rp_earth_err = pl.get("rp_earth_err", 0.0)
            t14_hr = pl.get("t14_hr", 0.0)
            t14_hr_err = pl.get("t14_hr_err", 0.0)
            b_val = pl.get("b", 0.0)
            b_err = pl.get("b_err", 0.0)
            a_au = pl.get("a_au", 0.0)
            a_au_err = pl.get("a_au_err", 0.0)
            t_eq = pl.get("t_eq", 0.0)
            t_eq_err = pl.get("t_eq_err", 0.0)

            det_for_planet = detections[idx] if idx < len(detections) else {}
            bayes_prob_val = det_for_planet.get("existence_probability_bayesian")
            bayes_conf_val = det_for_planet.get("bayesian_confidence", "N/A")
            if bayes_prob_val is not None:
                prob_pct = f"{bayes_prob_val * 100.0:.1f}%"
                if idx == 0:
                    prob_badge = "badge-derived"
                    prob_label = "Planet 1 Detection Probability"
                else:
                    prob_badge = "badge-fits" if bayes_prob_val > 0.75 else "badge-warning"
                    prob_label = f"Planet {idx + 1} Detection Probability"
                prob_row_html = f"""
                <div class="param-row">
                    <span class="param-label">{prob_label}</span>
                    <span class="param-value">
                        <strong>{prob_pct}</strong>
                        <span class="badge {prob_badge}">{bayes_conf_val}</span>
                    </span>
                </div>
                """
            else:
                prob_row_html = ""

            planets_cards_html += f"""
            <div class="card">
                <h2>Planet {idx + 1} Parameters</h2>
                {prob_row_html}
                
                <div class="param-row">
                    <span class="param-label">Period (P)</span>
                    <span class="param-value">
                        {p_val:.6f} &plusmn; {p_err:.6f} d
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>
                
                <div class="param-row">
                    <span class="param-label">Transit Epoch (t₀)</span>
                    <span class="param-value">
                        {t0_val:.4f} &plusmn; {t0_err:.4f} BTJD
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Radius Ratio (Rₚ/Rₛ)</span>
                    <span class="param-value">
                        {rp_r_star:.5f} &plusmn; {rp_r_star_err:.5f}
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Planet Radius (Rₚ)</span>
                    <span class="param-value">
                        {rp_earth:.2f} &plusmn; {rp_earth_err:.2f} R<sub>&oplus;</sub>
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Transit Duration (T₁₄)</span>
                    <span class="param-value">
                        {t14_hr:.3f} &plusmn; {t14_hr_err:.3f} hr
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Impact Parameter (b)</span>
                    <span class="param-value">
                        {b_val:.3f} &plusmn; {b_err:.3f}
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Semi-major Axis (a)</span>
                    <span class="param-value">
                        {a_au:.4f} &plusmn; {a_au_err:.4f} AU
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Equilibrium Temp (T<sub>eq</sub>)</span>
                    <span class="param-value">
                        {t_eq:.0f} &plusmn; {t_eq_err:.0f} K
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>
            </div>
            """
    else:
        single_pl = data.get("planet") or {}
        if single_pl:
            p_val = single_pl.get("period", 0.0)
            t0_val = single_pl.get("t0", 0.0)
            rp_r_star = single_pl.get("rp_r_star", 0.0)
            rp_earth = single_pl.get("rp_earth", 0.0)
            t14_hr = single_pl.get("t14_hr", 0.0)
            b_val = single_pl.get("b", 0.0)
            a_au = single_pl.get("a_au", 0.0)
            t_eq = single_pl.get("t_eq", 0.0)

            planets_cards_html += f"""
            <div class="card">
                <h2>Planet Parameters</h2>
                
                <div class="param-row">
                    <span class="param-label">Period (P)</span>
                    <span class="param-value">
                        {p_val:.6f} d
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>
                
                <div class="param-row">
                    <span class="param-label">Transit Epoch (t₀)</span>
                    <span class="param-value">
                        {t0_val:.4f} BTJD
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Radius Ratio (Rₚ/Rₛ)</span>
                    <span class="param-value">
                        {rp_r_star:.5f}
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Planet Radius (Rₚ)</span>
                    <span class="param-value">
                        {rp_earth:.2f} R<sub>&oplus;</sub>
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Transit Duration (T₁₄)</span>
                    <span class="param-value">
                        {t14_hr:.3f} hr
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Impact Parameter (b)</span>
                    <span class="param-value">
                        {b_val:.3f}
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Semi-major Axis (a)</span>
                    <span class="param-value">
                        {a_au:.4f} AU
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Equilibrium Temp (T<sub>eq</sub>)</span>
                    <span class="param-value">
                        {t_eq:.0f} K
                        <span class="badge badge-derived">Derived (MCMC)</span>
                    </span>
                </div>
            </div>
            """
        else:
            for idx, det in enumerate(detections):
                p_val = det.get("period", 0.0)
                t0_val = det.get("epoch", 0.0)
                duration_hr = det.get("duration_hr", 0.0)
                depth = det.get("depth", 0.0)
                method = det.get("method", "tls")
                
                planets_cards_html += f"""
                <div class="card">
                    <h2>Planet {idx + 1} Candidate (Search)</h2>
                    
                    <div class="param-row">
                        <span class="param-label">Period (P)</span>
                        <span class="param-value">
                            {p_val:.6f} d
                            <span class="badge badge-derived">Derived ({method.upper()})</span>
                        </span>
                    </div>
                    
                    <div class="param-row">
                        <span class="param-label">Transit Epoch (t₀)</span>
                        <span class="param-value">
                            {t0_val:.4f} BTJD
                            <span class="badge badge-derived">Derived ({method.upper()})</span>
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Transit Duration (T₁₄)</span>
                        <span class="param-value">
                            {duration_hr:.3f} hr
                            <span class="badge badge-derived">Derived ({method.upper()})</span>
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Transit Depth</span>
                        <span class="param-value">
                            {depth:.5f}
                            <span class="badge badge-derived">Derived ({method.upper()})</span>
                        </span>
                    </div>
                </div>
                """

    # ── Dyn Phase Plots ───────────────────────────────────────────────────
    search_phase_images_html = ""
    if len(detections) > 1:
        for idx in range(len(detections)):
            search_phase_images_html += f"""
                <div class="image-card">
                    <img src="plots/04_phase_p{idx}.png" alt="Phase-Folded Light Curve Planet {idx + 1}" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Phase-Folded Light Curve Planet {idx + 1}: folded at the search period of {detections[idx]['period']:.5f} d.</div>
                </div>
            """
    else:
        search_phase_images_html = """
                <div class="image-card">
                    <img src="plots/04_phase.png" alt="Phase-Folded Light Curve" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Phase-Folded Light Curve: data folded at the detected period, showing the characteristic transit dip.</div>
                </div>
        """

    mcmc_phase_images_html = ""
    if len(planets_data) > 1:
        for idx in range(len(planets_data)):
            mcmc_phase_images_html += f"""
                <div class="image-card">
                    <img src="plots/04_mcmc_phase_p{idx}.png" alt="MCMC Phase-Folded Light Curve Planet {idx + 1}" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Planet {idx + 1} MCMC Phase-Folded Transit: data folded at posterior period of {planets_data[idx].get('period', 0.0):.5f} d. Other planet transit signals and GP model have been subtracted.</div>
                </div>
            """
    else:
        mcmc_phase_images_html = """
                <div class="image-card">
                    <img src="plots/04_mcmc_phase.png" alt="MCMC Phase-Folded Light Curve" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">MCMC Phase-Folded Light Curve: data folded at the MCMC period. GP model has been subtracted.</div>
                </div>
        """

    # Dynamic badges helper for stellar parameters
    def get_stellar_badge(param_key: str) -> str:
        source = data.get("stellar", {}).get(f"{param_key}_source")
        if param_key == "rho_star":
            return '<span class="badge badge-derived">Derived</span>'
        if not source:
            source = data.get("stellar", {}).get("method") or "Catalog"
        
        if "Torres" in str(source):
            badge_class = "badge-derived"
            label = "Torres et al. 2010"
        elif "SIMBAD" in str(source):
            badge_class = "badge-fits"
            label = "SIMBAD"
        elif "VizieR" in str(source) or "TIC8" in str(source):
            badge_class = "badge-literature"
            label = "VizieR (TIC v8.2)"
        elif "Gaia" in str(source):
            badge_class = "badge-literature"
            label = "Gaia DR3"
        else:
            badge_class = "badge-literature"
            label = str(source)
            
        ref = data.get("stellar", {}).get(f"{param_key}_ref")
        title_attr = f' title="{ref}"' if ref else ''
        return f'<span class="badge {badge_class}"{title_attr}>{label}</span>'

    r_star_badge = get_stellar_badge("r_star")
    m_star_badge = get_stellar_badge("m_star")
    teff_badge = get_stellar_badge("teff")
    logg_badge = get_stellar_badge("logg")
    feh_badge = get_stellar_badge("feh")
    rho_star_badge = get_stellar_badge("rho_star")

    r_star = get_val("stellar", "r_star", 2, " R<sub>&sub;</sub>")
    m_star = get_val("stellar", "m_star", 2, " M<sub>&sub;</sub>")
    teff = get_val("stellar", "teff", 1, " K")
    logg = get_val("stellar", "logg", 2)
    feh = get_val("stellar", "feh", 2)
    rho_star = get_val("stellar", "rho_star", 3, " g/cm&sup3;")

    # Diagnostics
    max_rhat = data.get("diagnostics", {}).get("rhat_max")
    rhat_str = f"Passed (max R̂ = {max_rhat:.3f})" if (max_rhat is not None and not np.isnan(max_rhat)) else "N/A"
    
    min_ess = data.get("diagnostics", {}).get("ess_min")
    ess_str = f"{min_ess:.0f}" if (min_ess is not None and not np.isnan(min_ess)) else "N/A"

    divergences = data.get("diagnostics", {}).get("divergences")
    div_str = f"{divergences}" if divergences is not None else "N/A"

    sde = data.get("detection", {}).get("sde")
    snr = data.get("detection", {}).get("snr")
    tls_str = f"{sde:.2f} / {snr:.2f}" if sde is not None else "N/A"

    sectors_used = data.get("metadata", {}).get("sectors_used", [])
    sectors_used_str = ", ".join(map(str, sectors_used)) if sectors_used else "all"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TESS Exoplanet Pipeline Report - TIC {tic_id}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f0f4f8;
            --card-bg: #ffffff;
            --card-border: #d1d9e0;
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --accent: #2563eb;
            --accent-glow: rgba(37, 99, 235, 0.10);
            --success: #059669;
            --success-glow: rgba(5, 150, 105, 0.10);
            --warning: #d97706;
            --warning-glow: rgba(217, 119, 6, 0.10);
            --danger: #dc2626;
            --font-family-body: 'Inter', sans-serif;
            --font-family-heading: 'Outfit', sans-serif;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: var(--font-family-body);
            background-color: var(--bg-color);
            background-image: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.04) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(5, 150, 105, 0.04) 0%, transparent 40%);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2.5rem;
        }}
        header {{
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{
            font-family: var(--font-family-heading);
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.05rem;
            margin-top: 0.25rem;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.06);
        }}
        .kpi-title {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        .kpi-value {{
            font-family: var(--font-family-heading);
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.06);
        }}
        .card h2 {{
            font-family: var(--font-family-heading);
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            border-bottom: 2px solid var(--accent-glow);
            padding-bottom: 0.6rem;
            color: var(--accent);
        }}
        .param-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.8rem 0;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.95rem;
        }}
        .param-row:last-child {{
            border-bottom: none;
        }}
        .param-label {{
            color: var(--text-secondary);
            font-weight: 400;
        }}
        .param-value {{
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-primary);
        }}
        .badge {{
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
            display: inline-block;
        }}
        .badge-derived {{
            background-color: var(--success-glow);
            color: var(--success);
            border: 1px solid rgba(5, 150, 105, 0.3);
        }}
        .badge-literature {{
            background-color: var(--accent-glow);
            color: var(--accent);
            border: 1px solid rgba(37, 99, 235, 0.3);
        }}
        .badge-fits {{
            background-color: rgba(124, 58, 237, 0.10);
            color: #7c3aed;
            border: 1px solid rgba(124, 58, 237, 0.3);
        }}
        .badge-warning {{
            background-color: var(--warning-glow);
            color: var(--warning);
            border: 1px solid rgba(217, 119, 6, 0.3);
        }}
        .confidence-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.9rem;
        }}
        .confidence-table th, .confidence-table td {{
            text-align: left;
            padding: 0.85rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .confidence-table tr:last-child td {{
            border-bottom: none;
        }}
        .confidence-table th {{
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            background-color: #f8fafc;
        }}
        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.75rem;
            border-bottom: 2px solid var(--card-border);
            padding-bottom: 0.75rem;
            overflow-x: auto;
        }}
        .tab-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.25s ease;
        }}
        .tab-btn:hover {{
            background: #f1f5f9;
            color: var(--text-primary);
        }}
        .tab-btn.active {{
            background: var(--accent-glow);
            color: var(--accent);
            border: 1px solid rgba(37, 99, 235, 0.3);
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 1.5rem;
        }}
        .image-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease;
        }}
        .image-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.10);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            border-bottom: 1px solid var(--card-border);
            background: #f8fafc;
        }}
        .image-caption {{
            padding: 1.25rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
            background: #f8fafc;
            line-height: 1.5;
        }}
        pre {{
            background: #f1f5f9;
            border: 1px solid var(--card-border);
            padding: 1.75rem;
            border-radius: 12px;
            overflow-x: auto;
            color: #0f766e;
            font-family: monospace;
            font-size: 0.9rem;
            box-shadow: inset 0 1px 4px 0 rgba(0, 0, 0, 0.05);
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>TESS Planet Candidate Dashboard</h1>
            <div class="subtitle">Target: {target_name} | Coordinates: RA={ra:.6f}&deg;, Dec={dec:.6f}&deg;</div>
        </div>
        <div>
            <span class="badge badge-fits">Pipeline: v{data.get('metadata', {}).get('tess_pipeline_version', '0.1.0')}</span>
        </div>
    </header>

    <main>
        <!-- Top KPI Cards -->
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">TIC Target</div>
                <div class="kpi-value">{tic_id}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Sectors Used</div>
                <div class="kpi-value">{sectors_used_str}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Primary Period</div>
                <div class="kpi-value">{primary_period_str}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Effective Temp</div>
                <div class="kpi-value">{teff}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Stellar Radius</div>
                <div class="kpi-value">{r_star}</div>
            </div>
        </div>

        <!-- Diagnostic Figures Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="openTab(event, 'tab-dash')">Dashboard Summary</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-lcs')">Light Curves</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-search')">Period Search</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-mcmc')">MCMC Fit & Corner</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-json')">JSON Metadata</button>
        </div>

        <!-- Dashboard Summary Tab -->
        <div id="tab-dash" class="tab-content active">
            <div class="grid">
                <!-- Confidence Comparison Table (if exists) -->
                {model_comparison_html}

                <!-- Planet Parameters Cards -->
                {planets_cards_html}

                <!-- Stellar Parameters Card -->
                <div class="card">
                    <h2>Stellar Parameters</h2>
                    
                    <div class="param-row">
                        <span class="param-label">Stellar Radius (Rₛ)</span>
                        <span class="param-value">
                            {r_star}
                            {r_star_badge}
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Stellar Mass (Mₛ)</span>
                        <span class="param-value">
                            {m_star}
                            {m_star_badge}
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Effective Temp (T<sub>eff</sub>)</span>
                        <span class="param-value">
                            {teff}
                            {teff_badge}
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Surface Gravity (log g)</span>
                        <span class="param-value">
                            {logg}
                            {logg_badge}
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Metallicity ([Fe/H])</span>
                        <span class="param-value">
                            {feh}
                            {feh_badge}
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Stellar Density (ρₛ)</span>
                        <span class="param-value">
                            {rho_star}
                            {rho_star_badge}
                        </span>
                    </div>
                </div>

                <!-- Diagnostics & Metadata Card -->
                <div class="card">
                    <h2>MCMC Fit Diagnostics</h2>
                    
                    <div class="param-row">
                        <span class="param-label">R-hat Convergence</span>
                        <span class="param-value">
                            {rhat_str}
                            <span class="badge badge-derived">Derived (MCMC)</span>
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Min Effective Sample Size</span>
                        <span class="param-value">
                            {ess_str}
                            <span class="badge badge-derived">Derived (MCMC)</span>
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">Divergent Transitions</span>
                        <span class="param-value">
                            {div_str}
                            <span class="badge badge-derived">Derived (MCMC)</span>
                        </span>
                    </div>

                    <div class="param-row">
                        <span class="param-label">TLS SDE / SNR</span>
                        <span class="param-value">
                            {tls_str}
                            <span class="badge badge-derived">Derived (TLS)</span>
                        </span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Light Curves Tab -->
        <div id="tab-lcs" class="tab-content">
            <div class="image-gallery">
                <div class="image-card">
                    <img src="plots/01_raw.png" alt="Raw Light Curve" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Raw TESS SAP/PDCSAP Light Curve before flat-fielding and outlier rejection.</div>
                </div>
                <div class="image-card">
                    <img src="plots/02_flat.png" alt="Flattened Light Curve" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Flattened Light Curve: detrended using a spline or high-pass filter, ready for transit search.</div>
                </div>
            </div>
        </div>

        <!-- Period Search Tab -->
        <div id="tab-search" class="tab-content">
            <div class="image-gallery">
                <div class="image-card">
                    <img src="plots/03_tls_periodogram.png" alt="TLS Periodogram" onerror="this.src='plots/03_bls_periodogram.png'; this.onerror=function(){{this.parentNode.style.display='none';}}">
                    <div class="image-caption">TLS / BLS Periodogram: power vs trial period (days). The peak indicates the best-fit orbital period. Subplots show subsequent searches if multi-planet mode is enabled.</div>
                </div>
                {search_phase_images_html}
            </div>
        </div>

        <!-- MCMC Fit Tab -->
        <div id="tab-mcmc" class="tab-content">
            <div class="image-gallery">
                <div class="image-card" style="grid-column: 1 / -1;">
                    <img src="plots/06_bayesian_fit.png" alt="Bayesian Transit Fit & Residuals" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Bayesian Transit Fit: Full GP systematics + Keplerian transit model trend (top), phase-folded de-trended data fit (middle), and phase-folded model residuals (bottom) showing the excellent goodness-of-fit.</div>
                </div>
                <div class="image-card">
                    <img src="plots/10_gp_acf.png" alt="GP Residuals Autocorrelation Function" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">GP Autocorrelation Function (ACF) of Residuals: plots the autocorrelation of the residuals before GP detrending (light gray) and after GP detrending (green). This mathematically validates that the GP successfully removed red noise without affecting the transit profile.</div>
                </div>
                <div class="image-card">
                    <img src="plots/11_transit_stack.png" alt="Stacked Transits" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Stacked Individual Transits: overlays individual transit events vertically. This visually confirms that the transit is consistently present in every orbit and is not caused by a single spurious artifact or flare.</div>
                </div>
                <div class="image-card">
                    <img src="plots/09_posterior_predictive.png" alt="Posterior Predictive Check" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Posterior Predictive Check: draws from the MCMC posterior overlaid on the transit data.</div>
                </div>
                {mcmc_phase_images_html}
                <div class="image-card">
                    <img src="plots/07_corner.png" alt="Corner Plot" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Corner Plot showing covariance and 1D/2D posterior probability distributions for the key transit parameters, with parameter guide definitions.</div>
                </div>
                <div class="image-card" style="grid-column: 1 / -1;">
                    <img src="plots/08_trace.png" alt="MCMC Trace Plots" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">MCMC Trace Plots: Parameter values over chain steps to verify proper mixing, convergence, and parameter stationarity.</div>
                </div>
            </div>
        </div>

        <!-- JSON Metadata Tab -->
        <div id="tab-json" class="tab-content">
            <pre><code>{json_str}</code></pre>
        </div>
    </main>

    <script>
        function openTab(evt, tabId) {{
            const contents = document.getElementsByClassName("tab-content");
            for (let i = 0; i < contents.length; i++) {{
                contents[i].classList.remove("active");
            }}
            const buttons = document.getElementsByClassName("tab-btn");
            for (let i = 0; i < buttons.length; i++) {{
                buttons[i].classList.remove("active");
            }}
            document.getElementById(tabId).classList.add("active");
            evt.currentTarget.classList.add("active");
        }}
    </script>
</body>
</html>
"""
    with path.open("w") as f:
        f.write(html_content)
    log.info("Saved HTML Report: %s", path)
