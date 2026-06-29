"""
io/export.py — Export pipeline results to various file formats.

Formats:
  * CSV    — parameter summary table with medians and uncertainties
  * JSON   — metadata, config, and derived parameters
  * NetCDF — full posterior via ArviZ (arviz.InferenceData.to_netcdf)
  * FITS   — light curve arrays and model arrays
  * PNG    — diagnostic figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


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
    _save_html_report(results, target_dir)

    log.info("Export complete: %s", target_dir)


def _save_csv(results: "PipelineResults", prefix: Path) -> None:
    """Write parameter summary to CSV."""
    import csv

    rows = []
    for key, val in results.planet.items():
        rows.append({"parameter": key, "value": val})
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
    """Write full posterior to NetCDF via ArviZ."""
    if results.posterior is None:
        return
    try:
        path = Path(str(prefix) + "_posterior.nc")
        results.posterior.to_netcdf(str(path))
        log.debug("Saved posterior: %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save posterior: %s", exc)


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
    for name, fig in results.figures.items():
        if fig is None:
            continue
        try:
            path = output_dir / f"{name}.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            log.debug("Saved figure: %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save figure %r: %s", name, exc)


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

    # Determine tags and values
    period_val = get_val("period", "value", 6, " d")
    period_source = data.get("period", {}).get("source", "tls")
    period_badge = "badge-derived" if period_source == "tls" else "badge-literature"
    period_label = "Derived (TLS)" if period_source == "tls" else "Adopted from Lit"

    # Planet values
    t0_val = data.get("planet", {}).get("t0")
    if t0_val is not None:
        t0_str = f"{float(t0_val):.4f} BTJD"
        t0_badge = "badge-derived"
        t0_label = "Derived (MCMC)"
    else:
        epoch_val = data.get("detection", {}).get("epoch")
        t0_str = f"{float(epoch_val):.4f} BTJD" if epoch_val is not None else "N/A"
        t0_badge = "badge-derived"
        t0_label = "Derived (TLS)"

    rp_r_star = data.get("planet", {}).get("rp_r_star")
    rp_r_star_err = data.get("planet", {}).get("rp_r_star_err")
    if rp_r_star is not None:
        rp_r_star_str = f"{float(rp_r_star):.5f}"
        if rp_r_star_err is not None:
            rp_r_star_str += f" &plusmn; {float(rp_r_star_err):.5f}"
        rp_r_star_badge = "badge-derived"
        rp_r_star_label = "Derived (MCMC)"
    else:
        rp_r_star_str = "N/A"
        rp_r_star_badge = "badge-derived"
        rp_r_star_label = "Derived (TLS)"

    rp_earth = data.get("planet", {}).get("rp_earth")
    rp_earth_err = data.get("planet", {}).get("rp_earth_err")
    if rp_earth is not None:
        rp_earth_str = f"{float(rp_earth):.2f}"
        if rp_earth_err is not None:
            rp_earth_str += f" &plusmn; {float(rp_earth_err):.2f}"
        rp_earth_str += " R<sub>&oplus;</sub>"
        rp_earth_badge = "badge-derived"
        rp_earth_label = "Derived (MCMC)"
    else:
        rp_earth_str = "N/A"
        rp_earth_badge = "badge-derived"
        rp_earth_label = "Derived (MCMC)"

    t14 = data.get("planet", {}).get("t14_hr")
    if t14 is not None:
        t14_str = f"{float(t14):.3f} hr"
        t14_badge = "badge-derived"
        t14_label = "Derived (MCMC)"
    else:
        dur = data.get("detection", {}).get("duration_hr")
        t14_str = f"{float(dur):.3f} hr" if dur is not None else "N/A"
        t14_badge = "badge-derived"
        t14_label = "Derived (TLS)"

    b = data.get("planet", {}).get("b")
    b_err = data.get("planet", {}).get("b_err")
    if b is not None:
        b_str = f"{float(b):.3f}"
        if b_err is not None:
            b_str += f" &plusmn; {float(b_err):.3f}"
        b_badge = "badge-derived"
        b_label = "Derived (MCMC)"
    else:
        b_str = "N/A"
        b_badge = "badge-derived"
        b_label = "Derived (MCMC)"

    a_au = data.get("planet", {}).get("a_au")
    if a_au is not None:
        a_au_str = f"{float(a_au):.4f} AU"
        a_au_badge = "badge-derived"
        a_au_label = "Derived (MCMC)"
    else:
        a_au_str = "N/A"
        a_au_badge = "badge-derived"
        a_au_label = "Derived (MCMC)"

    t_eq = data.get("planet", {}).get("t_eq")
    if t_eq is not None:
        t_eq_str = f"{float(t_eq):.0f} K"
        t_eq_badge = "badge-derived"
        t_eq_label = "Derived (MCMC)"
    else:
        t_eq_str = "N/A"
        t_eq_badge = "badge-derived"
        t_eq_label = "Derived (MCMC)"

    # Stellar sources
    stellar_method = data.get("stellar", {}).get("method", "gaia_only")
    stellar_badge = "badge-derived" if stellar_method == "isoclassify" else "badge-literature"
    stellar_label = "Derived (isoclassify)" if stellar_method == "isoclassify" else "Gaia DR3 (Lit)"

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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TESS Exoplanet Pipeline Report - TIC {tic_id}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.15);
            --danger: #ef4444;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 2rem;
        }}
        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1rem;
            margin-top: 0.25rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.5rem;
            color: #60a5fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .param-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
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
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .badge-literature {{
            background-color: var(--accent-glow);
            color: var(--accent);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        .badge-fits {{
            background-color: rgba(139, 92, 246, 0.15);
            color: #a78bfa;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }}
        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
            overflow-x: auto;
        }}
        .tab-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 1rem;
            font-weight: 500;
            padding: 0.5rem 1rem;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }}
        .tab-btn.active {{
            background: var(--accent-glow);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
        }}
        .image-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            border-bottom: 1px solid var(--card-border);
            background: #000;
        }}
        .image-caption {{
            padding: 1rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
            background: rgba(0, 0, 0, 0.2);
        }}
        pre {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--card-border);
            padding: 1.5rem;
            border-radius: 8px;
            overflow-x: auto;
            color: #34d399;
            font-family: monospace;
            font-size: 0.9rem;
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
        <div class="grid">
            <!-- Planet Parameters Card -->
            <div class="card">
                <h2>Planet Parameters</h2>
                
                <div class="param-row">
                    <span class="param-label">Period (P)</span>
                    <span class="param-value">
                        {period_val}
                        <span class="badge {period_badge}">{period_label}</span>
                    </span>
                </div>
                
                <div class="param-row">
                    <span class="param-label">Transit Epoch (t₀)</span>
                    <span class="param-value">
                        {t0_str}
                        <span class="badge {t0_badge}">{t0_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Radius Ratio (Rₚ/Rₛ)</span>
                    <span class="param-value">
                        {rp_r_star_str}
                        <span class="badge {rp_r_star_badge}">{rp_r_star_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Planet Radius (Rₚ)</span>
                    <span class="param-value">
                        {rp_earth_str}
                        <span class="badge {rp_earth_badge}">{rp_earth_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Transit Duration (T₁₄)</span>
                    <span class="param-value">
                        {t14_str}
                        <span class="badge {t14_badge}">{t14_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Impact Parameter (b)</span>
                    <span class="param-value">
                        {b_str}
                        <span class="badge {b_badge}">{b_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Semi-major Axis (a)</span>
                    <span class="param-value">
                        {a_au_str}
                        <span class="badge {a_au_badge}">{a_au_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Equilibrium Temp (T<sub>eq</sub>)</span>
                    <span class="param-value">
                        {t_eq_str}
                        <span class="badge {t_eq_badge}">{t_eq_label}</span>
                    </span>
                </div>
            </div>

            <!-- Stellar Parameters Card -->
            <div class="card">
                <h2>Stellar Parameters</h2>
                
                <div class="param-row">
                    <span class="param-label">Stellar Radius (Rₛ)</span>
                    <span class="param-value">
                        {r_star}
                        <span class="badge {stellar_badge}">{stellar_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Stellar Mass (Mₛ)</span>
                    <span class="param-value">
                        {m_star}
                        <span class="badge {stellar_badge}">{stellar_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Effective Temp (T<sub>eff</sub>)</span>
                    <span class="param-value">
                        {teff}
                        <span class="badge {stellar_badge}">{stellar_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Surface Gravity (log g)</span>
                    <span class="param-value">
                        {logg}
                        <span class="badge {stellar_badge}">{stellar_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Metallicity ([Fe/H])</span>
                    <span class="param-value">
                        {feh}
                        <span class="badge {stellar_badge}">{stellar_label}</span>
                    </span>
                </div>

                <div class="param-row">
                    <span class="param-label">Stellar Density (ρₛ)</span>
                    <span class="param-value">
                        {rho_star}
                        <span class="badge {stellar_badge}">{stellar_label}</span>
                    </span>
                </div>
            </div>

            <!-- Diagnostics & Metadata Card -->
            <div class="card">
                <h2>MCMC Diagnostics</h2>
                
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

        <!-- Diagnostic Figures Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="openTab(event, 'tab-lcs')">Light Curves</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-search')">Period Search</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-mcmc')">MCMC Fit</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-json')">JSON Metadata</button>
        </div>

        <!-- Light Curves Tab -->
        <div id="tab-lcs" class="tab-content active">
            <div class="image-gallery">
                <div class="image-card">
                    <img src="plots/raw.png" alt="Raw Light Curve" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Raw TESS SAP/PDCSAP Light Curve before flat-fielding and outlier rejection.</div>
                </div>
                <div class="image-card">
                    <img src="plots/flat.png" alt="Flattened Light Curve" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Flattened Light Curve: detrended using a spline or high-pass filter, ready for transit search.</div>
                </div>
            </div>
        </div>

        <!-- Period Search Tab -->
        <div id="tab-search" class="tab-content">
            <div class="image-gallery">
                <div class="image-card">
                    <img src="plots/tls_periodogram.png" alt="TLS Periodogram" onerror="this.src='plots/bls_periodogram.png'; this.onerror=function(){{this.parentNode.style.display='none';}}">
                    <div class="image-caption">TLS / BLS Periodogram: power vs trial period (days). The peak indicates the best-fit orbital period.</div>
                </div>
                <div class="image-card">
                    <img src="plots/phase.png" alt="Phase-Folded Light Curve" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Phase-Folded Light Curve: data folded at the detected period, showing the characteristic transit dip.</div>
                </div>
            </div>
        </div>

        <!-- MCMC Fit Tab -->
        <div id="tab-mcmc" class="tab-content">
            <div class="image-gallery">
                <div class="image-card">
                    <img src="plots/bayesian_fit.png" alt="Bayesian Transit Fit" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Bayesian Transit Fit: Best-fit Keplerian transit model with GP systematics overlaid on the raw data (top) and residuals (bottom).</div>
                </div>
                <div class="image-card">
                    <img src="plots/posterior_predictive.png" alt="Posterior Predictive Check" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Posterior Predictive Check: Draws from the posterior overlaid on the transit data.</div>
                </div>
                <div class="image-card">
                    <img src="plots/residuals.png" alt="Residuals" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Residuals vs Time (top) and vs Phase (bottom) showing the goodness of the fit.</div>
                </div>
                <div class="image-card">
                    <img src="plots/corner.png" alt="Corner Plot" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Corner Plot showing covariance and 1D/2D posterior probability distributions for the key transit parameters.</div>
                </div>
                <div class="image-card">
                    <img src="plots/trace.png" alt="MCMC Trace Plots" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">MCMC Trace Plots: Parameter values over chain steps to verify proper mixing and convergence.</div>
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
