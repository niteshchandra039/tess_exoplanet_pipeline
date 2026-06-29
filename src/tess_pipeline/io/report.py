"""
io/report.py — PDF summary report generation.

Generates a single-page PDF with:
  * Parameter table (median + 16/84% credible intervals)
  * Key figures (phase-fold, corner plot snippet, convergence table)
  * Convergence diagnostics table

Requires matplotlib's PDF backend (always available).
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

    with PdfPages(str(pdf_path)) as pdf:
        # ── Page 1: Summary + phase-fold ──────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # Title
        fig.suptitle(
            f"TESS Pipeline Report — TIC {tic}",
            fontsize=14,
            fontweight="bold",
            y=0.97,
        )

        # Parameter table (top-left)
        ax_table = fig.add_subplot(gs[0, 0])
        ax_table.axis("off")
        table_data = _build_table_data(results)
        if table_data:
            col_labels = ["Parameter", "Value"]
            t = ax_table.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1.0, 1.4)
        ax_table.set_title("Derived Parameters", fontsize=9, pad=4)

        # Phase-fold (top-right)
        ax_phase = fig.add_subplot(gs[0, 1])
        phase_fig = results.figures.get("phase")
        if phase_fig is not None:
            # Re-render into the subplot by extracting data
            _copy_figure_axes(phase_fig, ax_phase, title="Phase-Folded Transit")
        else:
            ax_phase.text(0.5, 0.5, "Phase plot unavailable", ha="center", va="center")
            ax_phase.set_title("Phase-Folded Transit", fontsize=9)

        # Convergence diagnostics (bottom-left)
        ax_diag = fig.add_subplot(gs[1, 0])
        ax_diag.axis("off")
        diag_data = _build_diagnostics_table(results)
        if diag_data:
            t2 = ax_diag.table(
                cellText=diag_data,
                colLabels=["Diagnostic", "Value"],
                loc="center",
                cellLoc="left",
            )
            t2.auto_set_font_size(False)
            t2.set_fontsize(8)
            t2.scale(1.0, 1.4)
        ax_diag.set_title("Convergence Diagnostics", fontsize=9, pad=4)

        # Stellar parameters (bottom-right)
        ax_stellar = fig.add_subplot(gs[1, 1])
        ax_stellar.axis("off")
        stellar_data = _build_stellar_table(results)
        if stellar_data:
            t3 = ax_stellar.table(
                cellText=stellar_data,
                colLabels=["Stellar Parameter", "Value"],
                loc="center",
                cellLoc="left",
            )
            t3.auto_set_font_size(False)
            t3.set_fontsize(8)
            t3.scale(1.0, 1.4)
        ax_stellar.set_title("Stellar Parameters", fontsize=9, pad=4)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Additional pages: figures ──────────────────────────────────────────
        for name in ("tls_periodogram", "bls_periodogram", "corner", "trace"):
            fig_stored = results.figures.get(name)
            if fig_stored is not None:
                pdf.savefig(fig_stored, bbox_inches="tight")

        # ── PDF metadata ──────────────────────────────────────────────────────
        d = pdf.infodict()
        d["Title"] = f"TESS Pipeline Report — TIC {tic}"
        d["Subject"] = "Exoplanet transit characterization"
        d["Creator"] = f"tess_pipeline v{results.metadata.get('tess_pipeline_version', '?')}"

    log.info("PDF report saved: %s", pdf_path)
    return pdf_path


def _build_table_data(results: "PipelineResults") -> list[list[str]]:
    pl = results.planet
    rows: list[list[str]] = []

    def _fmt(val: Any, err: Any = None) -> str:
        if val is None:
            return "N/A"
        s = f"{val:.4g}"
        if err is not None:
            s += f" ± {err:.3g}"
        return s

    period = results.period.get("value")
    rows.append(["Period (d)", _fmt(period)])
    rows.append(["Rp/R★", _fmt(pl.get("rp_r_star"), pl.get("rp_r_star_err"))])
    rows.append(["Rp (R⊕)", _fmt(pl.get("rp_earth"), pl.get("rp_earth_err"))])
    rows.append(["b", _fmt(pl.get("b"), pl.get("b_err"))])
    rows.append(["T14 (h)", _fmt(pl.get("t14_hr"), pl.get("t14_hr_err"))])
    rows.append(["a (AU)", _fmt(pl.get("a_au"), pl.get("a_au_err"))])
    rows.append(["a/R★", _fmt(pl.get("a_r_star"))])
    rows.append(["Teq (K)", _fmt(pl.get("t_eq"))])
    return rows


def _build_diagnostics_table(results: "PipelineResults") -> list[list[str]]:
    d = results.diagnostics
    rows: list[list[str]] = []
    if "rhat_max" in d:
        rows.append(["R̂ (max)", f"{d['rhat_max']:.4f}" if d["rhat_max"] else "N/A"])
    if "ess_min" in d:
        rows.append(["ESS (min)", f"{d['ess_min']:.0f}" if d["ess_min"] else "N/A"])
    if "divergences" in d:
        rows.append(["Divergences", str(d["divergences"])])
    if "converged" in d:
        rows.append(["Converged", "Yes" if d["converged"] else "No"])
    return rows


def _build_stellar_table(results: "PipelineResults") -> list[list[str]]:
    st = results.stellar
    rows: list[list[str]] = []

    def _fmt(val: Any, err: Any = None) -> str:
        if val is None:
            return "N/A"
        s = f"{val:.4g}"
        if err is not None:
            s += f" ± {err:.3g}"
        return s

    rows.append(["R★ (R☉)", _fmt(st.get("r_star"), st.get("r_star_err"))])
    rows.append(["M★ (M☉)", _fmt(st.get("m_star"), st.get("m_star_err"))])
    rows.append(["Teff (K)", _fmt(st.get("teff"), st.get("teff_err"))])
    rows.append(["log g (dex)", _fmt(st.get("logg"), st.get("logg_err"))])
    rows.append(["[Fe/H]", _fmt(st.get("feh"), st.get("feh_err"))])
    rows.append(["ρ★ (g/cm³)", _fmt(st.get("rho_star"), st.get("rho_star_err"))])
    rows.append(["Method", str(st.get("method", "N/A"))])
    return rows


def _copy_figure_axes(src_fig: Any, dst_ax: Any, title: str = "") -> None:
    """Attempt to copy the first axes content from src_fig into dst_ax."""
    import numpy as np

    try:
        src_ax = src_fig.axes[0]
        for line in src_ax.lines:
            dst_ax.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), linewidth=line.get_linewidth())
        for coll in src_ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                dst_ax.scatter(
                    offsets[:, 0], offsets[:, 1],
                    s=1.0, color="steelblue", alpha=0.4, rasterized=True,
                )
        dst_ax.set_xlabel(src_ax.get_xlabel(), fontsize=7)
        dst_ax.set_ylabel(src_ax.get_ylabel(), fontsize=7)
        dst_ax.tick_params(labelsize=7)
        if title:
            dst_ax.set_title(title, fontsize=9)
    except Exception:  # noqa: BLE001
        dst_ax.text(0.5, 0.5, title or "Figure unavailable", ha="center", va="center",
                    fontsize=8)
