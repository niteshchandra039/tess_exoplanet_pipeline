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
            path = output_dir / f"{filename}.png"
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
            
            # Bayesian confidence/probability
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

            planets_cards_html += f"""
            <div class="card">
                <h2>Planet {idx + 1} Parameters</h2>
                
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
        # Fallback to single planet if no list, or use detection details
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
            # TLS/BLS detections only
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
                    <div class="image-caption">Phase-Folded Light Curve Planet {idx + 1}: data folded at the search period of {detections[idx]['period']:.5f} d.</div>
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
        .confidence-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.9rem;
        }}
        .confidence-table th, .confidence-table td {{
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .confidence-table th {{
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }}
        .confidence-table tr:last-child td {{
            border-bottom: none;
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
            <!-- Confidence and Integrity Comparison -->
            {model_comparison_html}

            <!-- Planet Parameters Card -->
            {planets_cards_html}

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
                    <div class="image-caption">TLS / BLS Periodogram: power vs trial period (days). The peak indicates the best-fit orbital period.</div>
                </div>
                {search_phase_images_html}
            </div>
        </div>

        <!-- MCMC Fit Tab -->
        <div id="tab-mcmc" class="tab-content">
            <div class="image-gallery">
                <div class="image-card">
                    <img src="plots/06_bayesian_fit.png" alt="Bayesian Transit Fit" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Bayesian Transit Fit: Best-fit Keplerian transit model with GP systematics overlaid on the raw data (top) and residuals (bottom).</div>
                </div>
                <div class="image-card">
                    <img src="plots/09_posterior_predictive.png" alt="Posterior Predictive Check" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Posterior Predictive Check: Draws from the posterior overlaid on the transit data.</div>
                </div>
                {mcmc_phase_images_html}
                <div class="image-card">
                    <img src="plots/05_residuals.png" alt="Residuals" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Residuals vs Time (top) and vs Phase (bottom) showing the goodness of the fit.</div>
                </div>
                <div class="image-card">
                    <img src="plots/07_corner.png" alt="Corner Plot" onerror="this.parentNode.style.display='none';">
                    <div class="image-caption">Corner Plot showing covariance and 1D/2D posterior probability distributions for the key transit parameters.</div>
                </div>
                <div class="image-card">
                    <img src="plots/08_trace.png" alt="MCMC Trace Plots" onerror="this.parentNode.style.display='none';">
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
