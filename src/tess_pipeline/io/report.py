"""
io/report.py — PDF summary report generation.

Generates a publication-quality PDF report with:
  * Literature vs. Derived comparison table
  * Stellar host star parameters
  * Derived planet parameters (multi-candidate support)
  * MCMC convergence diagnostics
  * Key figures (phased transit fits, periodograms, GP ACF, trace/corner)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


def generate_pdf_report(results: "PipelineResults", output_dir: Path) -> Path:
    """
    Generate a PDF summary report for a pipeline run.

    Parameters
    ----------
    results : PipelineResults
    output_dir : Path
        Directory to write the PDF.

    Returns
    -------
    Path to the generated PDF file.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tic = results.target.get("tic_id", "unknown")
    pdf_path = output_dir / f"TIC{tic}_report.pdf"

    # Use a clean, professional color palette
    # Primary: #0f766e (Teal), Secondary: #2563eb (Blue)
    
    with PdfPages(str(pdf_path)) as pdf:
        # ── Page 1: Executive Summary ──────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # Main Header Banner
        ra = results.target.get("ra", 0.0)
        dec = results.target.get("dec", 0.0)
        sectors = results.metadata.get("sectors_used", [])
        sectors_str = ", ".join(map(str, sectors)) if sectors else "all"
        
        header_text = (
            f"TESS PLANET TRANSIT CHARACTERIZATION REPORT\n"
            f"Target: TIC {tic} | RA: {ra:.5f}° | Dec: {dec:.5f}° | Sectors: {sectors_str}"
        )
        fig.suptitle(header_text, fontsize=12, fontweight="bold", y=0.96, color="#0f766e")

        # 1. Derived Planet Parameters (Top-Left)
        ax_table = fig.add_subplot(gs[0, 0])
        ax_table.axis("off")
        table_data = _build_table_data(results)
        if table_data:
            col_labels = ["Parameter", "Derived Value (MCMC)"]
            t = ax_table.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            t.auto_set_font_size(False)
            t.set_fontsize(7.5)
            t.scale(1.0, 1.3)
            # Style header row
            for col_idx in range(len(col_labels)):
                cell = t[0, col_idx]
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#0f766e")
        ax_table.set_title("Derived Planetary Parameters", fontsize=9, fontweight="bold", pad=6, color="#1e293b")

        # 2. Primary Phase-Folded Transit Plot (Top-Right)
        ax_phase = fig.add_subplot(gs[0, 1])
        phase_fig = (
            results.figures.get("mcmc_phase")
            or results.figures.get("mcmc_phase_p0")
            or results.figures.get("phase")
            or results.figures.get("phase_p0")
        )
        if phase_fig is not None:
            _copy_figure_axes(phase_fig, ax_phase, title="Primary Phased Transit")
        else:
            ax_phase.text(0.5, 0.5, "Phased transit plot unavailable", ha="center", va="center", fontsize=8.5, color="#64748b")
            ax_phase.set_title("Primary Phased Transit", fontsize=9, fontweight="bold", color="#1e293b")

        # 3. Literature vs. MCMC Comparison & Diagnostics (Bottom-Left)
        ax_diag = fig.add_subplot(gs[1, 0])
        ax_diag.axis("off")
        diag_data = _build_comparison_and_diagnostics(results)
        if diag_data:
            col_labels = ["Parameter / Metric", "Literature Value", "Derived / Diagnostic"]
            t2 = ax_diag.table(
                cellText=diag_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            t2.auto_set_font_size(False)
            t2.set_fontsize(7.5)
            t2.scale(1.0, 1.3)
            for col_idx in range(len(col_labels)):
                cell = t2[0, col_idx]
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#0f766e")
        ax_diag.set_title("Literature Comparison & MCMC Diagnostics", fontsize=9, fontweight="bold", pad=6, color="#1e293b")

        # 4. Host Star Parameters (Bottom-Right)
        ax_stellar = fig.add_subplot(gs[1, 1])
        ax_stellar.axis("off")
        stellar_data = _build_stellar_table(results)
        if stellar_data:
            col_labels = ["Stellar Parameter", "Value"]
            t3 = ax_stellar.table(
                cellText=stellar_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            t3.auto_set_font_size(False)
            t3.set_fontsize(7.5)
            t3.scale(1.0, 1.3)
            for col_idx in range(len(col_labels)):
                cell = t3[0, col_idx]
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#0f766e")
        ax_stellar.set_title("Host Star Properties", fontsize=9, fontweight="bold", pad=6, color="#1e293b")

        # Add a footer label on page 1
        fig.text(
            0.5, 0.02,
            f"Generated automatically by TESS Exoplanet Analysis Pipeline. All uncertainties are 68% credible intervals.",
            fontsize=7.5, color="#64748b", ha="center"
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Pages 2+: Diagnostic Figures ───────────────────────────────────────
        # Specify preferred ordering of plots
        preferred_order = [
            "tls_periodogram",
            "bls_periodogram",
            "bayesian_fit",
            "gp_acf",
            "transit_stack",
            "mcmc_phase",
            "posterior_predictive",
            "corner",
            "trace"
        ]

        # Save preferred figures if available
        for name in preferred_order:
            fig_stored = results.figures.get(name)
            if fig_stored is not None:
                pdf.savefig(fig_stored, bbox_inches="tight")

        # Save any planet-specific phase/stack plots not in preferred_order (e.g. mcmc_phase_p1)
        for name, fig_stored in results.figures.items():
            if fig_stored is None:
                continue
            if name in preferred_order or name in ("raw", "flat", "phase", "residuals"):
                # Skip duplicate or raw plots to keep PDF concise unless they are periodograms/fits
                continue
            # Make sure it's a matplotlib figure
            if hasattr(fig_stored, "savefig"):
                pdf.savefig(fig_stored, bbox_inches="tight")

        # ── PDF metadata ──────────────────────────────────────────────────────
        d = pdf.infodict()
        d["Title"] = f"TESS Pipeline Report — TIC {tic}"
        d["Subject"] = "Exoplanet transit characterization & validation"
        d["Creator"] = f"tess_pipeline v{results.metadata.get('tess_pipeline_version', '0.1.0')}"

    log.info("PDF report saved: %s", pdf_path)
    return pdf_path


def _build_table_data(results: "PipelineResults") -> list[list[str]]:
    """Build derived planet parameters list."""
    rows: list[list[str]] = []

    def _fmt(val: Any, err: Any = None) -> str:
        if val is None or val == "":
            return "N/A"
        try:
            fval = float(val)
            s = f"{fval:.5g}"
            if err is not None:
                s += f" ± {float(err):.3g}"
            return s
        except Exception:
            return str(val)

    if results.planets:
        for idx, pl in enumerate(results.planets):
            p_prefix = f"P{idx+1} " if len(results.planets) > 1 else ""
            rows.append([f"{p_prefix}Period (d)", _fmt(pl.get("period"), pl.get("period_err"))])
            rows.append([f"{p_prefix}Rp/R★", _fmt(pl.get("rp_r_star"), pl.get("rp_r_star_err"))])
            rows.append([f"{p_prefix}Rp (R⊕)", _fmt(pl.get("rp_earth"), pl.get("rp_earth_err"))])
            rows.append([f"{p_prefix}b", _fmt(pl.get("b"), pl.get("b_err"))])
            rows.append([f"{p_prefix}T14 (h)", _fmt(pl.get("t14_hr"), pl.get("t14_hr_err"))])
            rows.append([f"{p_prefix}a (AU)", _fmt(pl.get("a_au"), pl.get("a_au_err"))])
            rows.append([f"{p_prefix}Teq (K)", _fmt(pl.get("t_eq"), pl.get("t_eq_err"))])
    else:
        pl = results.planet or {}
        period = results.period.get("value") if results.period else None
        rows.append(["Period (d)", _fmt(period or pl.get("period"))])
        rows.append(["Rp/R★", _fmt(pl.get("rp_r_star"), pl.get("rp_r_star_err"))])
        rows.append(["Rp (R⊕)", _fmt(pl.get("rp_earth"), pl.get("rp_earth_err"))])
        rows.append(["b", _fmt(pl.get("b"), pl.get("b_err"))])
        rows.append(["T14 (h)", _fmt(pl.get("t14_hr"), pl.get("t14_hr_err"))])
        rows.append(["a (AU)", _fmt(pl.get("a_au"), pl.get("a_au_err"))])
        rows.append(["a/R★", _fmt(pl.get("a_r_star"))])
        rows.append(["Teq (K)", _fmt(pl.get("t_eq"))])

    return rows


def _build_comparison_and_diagnostics(results: "PipelineResults") -> list[list[str]]:
    """Build comparison between literature and derived periods, plus MCMC convergence metrics."""
    rows: list[list[str]] = []

    def _fmt(val: Any) -> str:
        if val is None or val == "":
            return "N/A"
        try:
            return f"{float(val):.6g}"
        except Exception:
            return str(val)

    # 1. Lit Period & Epoch
    lit_period = results.metadata.get("archive_period")
    if lit_period is None and results.period and results.period.get("source") == "archive":
        lit_period = results.period.get("value")

    lit_epoch = results.metadata.get("archive_epoch")
    
    # Derived values
    if results.planets:
        p_val = results.planets[0].get("period")
        p_err = results.planets[0].get("period_err")
        t0_val = results.planets[0].get("t0")
        t0_err = results.planets[0].get("t0_err")
        
        p_derived_str = f"{p_val:.6f}" + (f" ± {p_err:.5f}" if p_err else "")
        t0_derived_str = f"{t0_val:.4f}" + (f" ± {t0_err:.4f}" if t0_err else "")
    else:
        pl = results.planet or {}
        p_val = pl.get("period") or (results.period.get("value") if results.period else None)
        p_err = pl.get("period_err")
        t0_val = pl.get("t0") or results.detection.get("epoch")
        t0_err = pl.get("t0_err")
        
        p_derived_str = f"{p_val:.6f}" if p_val else "N/A"
        if p_val and p_err:
            p_derived_str += f" ± {p_err:.5f}"
        t0_derived_str = f"{t0_val:.4f}" if t0_val else "N/A"
        if t0_val and t0_err:
            t0_derived_str += f" ± {t0_err:.4f}"

    rows.append(["P1 Period (d)", _fmt(lit_period), p_derived_str])
    rows.append(["P1 Epoch (BTJD)", _fmt(lit_epoch), t0_derived_str])

    # 2. Model comparison metrics
    comp = results.metadata.get("model_comparison") or {}
    delta_bic = comp.get("delta_bic")
    prob_multi = comp.get("probability_multiplanet")
    if delta_bic is not None:
        rows.append(["Model ΔBIC (2p - 1p)", "N/A", f"{delta_bic:.2f}"])
    if prob_multi is not None:
        rows.append(["Multiplanet Prob", "N/A", f"{prob_multi * 100.0:.2f}%"])

    # 3. MCMC Diagnostics
    d = results.diagnostics or {}
    rhat_max = d.get("rhat_max")
    ess_min = d.get("ess_min")
    divs = d.get("divergences")
    
    rows.append(["Max R-hat (R̂)", "N/A", f"{rhat_max:.4f}" if rhat_max else "N/A"])
    rows.append(["Min ESS", "N/A", f"{ess_min:.0f}" if ess_min else "N/A"])
    rows.append(["MCMC Divergences", "N/A", str(divs) if divs is not None else "N/A"])

    return rows


def _build_stellar_table(results: "PipelineResults") -> list[list[str]]:
    """Build stellar parameter table comparing literature vs. isoclassify/Gaia."""
    st = results.stellar
    rows: list[list[str]] = []

    def _fmt(val: Any, err: Any = None) -> str:
        if val is None or val == "":
            return "N/A"
        try:
            fval = float(val)
            s = f"{fval:.4g}"
            if err is not None:
                s += f" ± {float(err):.3g}"
            return s
        except Exception:
            return str(val)

    rows.append(["R★ (R☉)", _fmt(st.get("r_star"), st.get("r_star_err"))])
    rows.append(["M★ (M☉)", _fmt(st.get("m_star"), st.get("m_star_err"))])
    rows.append(["Teff (K)", _fmt(st.get("teff"), st.get("teff_err"))])
    rows.append(["log g (dex)", _fmt(st.get("logg"), st.get("logg_err"))])
    rows.append(["[Fe/H]", _fmt(st.get("feh"), st.get("feh_err"))])
    rows.append(["ρ★ (g/cm³)", _fmt(st.get("rho_star"), st.get("rho_star_err"))])
    rows.append(["Stellar Source", str(st.get("method", "Gaia DR3 (Lit)"))])
    return rows


def _copy_figure_axes(src_fig: Any, dst_ax: Any, title: str = "") -> None:
    """Attempt to copy the first axes content from src_fig into dst_ax."""
    import numpy as np

    try:
        # Find primary axes with actual plot lines/collections
        plot_ax = None
        for ax in src_fig.axes:
            if len(ax.lines) > 0 or len(ax.collections) > 0:
                plot_ax = ax
                break
        if plot_ax is None and len(src_fig.axes) > 0:
            plot_ax = src_fig.axes[0]
            
        if plot_ax is None:
            raise ValueError("No axes found")

        # Copy line plots
        for line in plot_ax.lines:
            dst_ax.plot(
                line.get_xdata(), line.get_ydata(),
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                alpha=line.get_alpha()
            )
            
        # Copy scatter plots / errorbars
        for coll in plot_ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                dst_ax.scatter(
                    offsets[:, 0], offsets[:, 1],
                    s=1.5,
                    color=coll.get_facecolors()[0] if len(coll.get_facecolors()) > 0 else "gray",
                    alpha=coll.get_alpha() or 0.3,
                    rasterized=True
                )
                
        # Copy labels and limits
        dst_ax.set_xlabel(plot_ax.get_xlabel(), fontsize=7.5)
        dst_ax.set_ylabel(plot_ax.get_ylabel(), fontsize=7.5)
        dst_ax.set_xlim(plot_ax.get_xlim())
        dst_ax.set_ylim(plot_ax.get_ylim())
        dst_ax.tick_params(labelsize=7)
        if title:
            dst_ax.set_title(title, fontsize=8.5, fontweight="bold", color="#1e293b")
            
    except Exception as exc:  # noqa: BLE001
        dst_ax.text(
            0.5, 0.5,
            f"Plot load error: {exc!s}",
            ha="center", va="center", fontsize=7.5, color="#ef4444"
        )
