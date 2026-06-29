"""
results.py — PipelineResults dataclass returned by Pipeline.run().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


@dataclass
class PipelineResults:
    """
    Container for all outputs produced by a pipeline run.

    Fields
    ------
    target      : dict with 'tic_id', 'ra', 'dec'
    lightcurve  : preprocessed lightkurve.LightCurve (or None)
    period      : {'value': float, 'source': 'archive'|'tls'|'bls'|'override'}
    detection   : TLS/BLS result dict (SDE/SNR, epoch, duration, depth, method)
    stellar     : Gaia + isoclassify parameters; None values for unavailable fields
    rv          : {'rv': float, 'rv_err': float} or None
    posterior   : arviz.InferenceData or None
    planet      : derived planetary parameters with credible intervals
    model       : {'time', 'flux_model', 'gp_model', 'residuals'}
    diagnostics : {'rhat': dict, 'ess': dict, 'divergences': int, 'ppc': dict}
    figures     : dict[str, matplotlib.figure.Figure]
    metadata    : run configuration, timestamps, package version
    """

    target: dict[str, Any] = field(default_factory=dict)
    lightcurve: Any = None
    period: dict[str, Any] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)
    stellar: dict[str, Any] = field(default_factory=dict)
    rv: dict[str, Any] | None = None
    posterior: Any = None
    planet: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a human-readable parameter summary."""
        lines = [
            "=" * 60,
            f"  TESS Pipeline Results — TIC {self.target.get('tic_id', '?')}",
            "=" * 60,
        ]

        # Period
        p = self.period
        if p:
            lines.append(
                f"  Period         : {p.get('value', '?'):.6f} d  [{p.get('source', '?')}]"
            )

        # Detection
        det = self.detection
        if det:
            lines.append(f"  Epoch (t0)     : {det.get('epoch', '?')}")
            lines.append(f"  Duration (T14) : {det.get('duration_hr', '?'):.3f} h")
            lines.append(f"  Depth          : {det.get('depth', '?'):.6f}")
            stat_key = "SDE" if "sde" in det else "SNR"
            lines.append(f"  Detection stat : {stat_key} = {det.get(stat_key.lower(), '?'):.1f}")

        # Stellar
        st = self.stellar
        if st:
            lines.append(f"  R★             : {st.get('r_star', '?')} R_Sun")
            lines.append(f"  M★             : {st.get('m_star', '?')} M_Sun")
            lines.append(f"  Teff           : {st.get('teff', '?')} K")
            lines.append(f"  ρ★             : {st.get('rho_star', '?')} g/cm³")

        # Planet (Bayesian)
        pl = self.planet
        if pl:
            lines.append("-" * 60)
            lines.append("  Bayesian planet parameters:")
            for key in ("rp_earth", "a_au", "a_r_star", "t_eq", "rp_r_star", "b", "t14_hr"):
                val = pl.get(key)
                err = pl.get(f"{key}_err")
                if val is not None:
                    err_str = f" ± {err:.4g}" if err is not None else ""
                    lines.append(f"    {key:<15}: {val:.4g}{err_str}")

        # Diagnostics
        diag = self.diagnostics
        if diag:
            lines.append("-" * 60)
            lines.append(f"  Divergences    : {diag.get('divergences', '?')}")
            rhat_max = diag.get("rhat_max")
            ess_min = diag.get("ess_min")
            if rhat_max is not None:
                lines.append(f"  R-hat (max)    : {rhat_max:.4f}")
            if ess_min is not None:
                lines.append(f"  ESS (min)      : {ess_min:.0f}")

        lines.append("=" * 60)
        print("\n".join(lines))

    # ── Plotting ─────────────────────────────────────────────────────────────

    _STAGE_MAP: ClassVar[dict[str, str]] = {
        "raw": "plot_raw_lightcurve",
        "flat": "plot_flattened_lightcurve",
        "periodogram": "plot_periodogram",
        "phase": "plot_phase_curve",
        "posterior": "plot_posterior",
        "residuals": "plot_residuals",
        "corner": "plot_corner",
        "trace": "plot_trace",
    }

    def plot(self, stage: str) -> Any:
        """Retrieve a pre-computed figure by stage name."""
        if stage not in self.figures:
            available = list(self.figures.keys())
            raise KeyError(
                f"No figure for stage {stage!r}. Available: {available}"
            )
        return self.figures[stage]

    def plot_all(self) -> None:
        """Display all available figures."""
        import matplotlib.pyplot as plt

        for name, fig in self.figures.items():
            if fig is not None:
                fig.suptitle(name, fontsize=10)
                plt.show()

    # ── Export ───────────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path = "output") -> None:
        """Write all outputs to *output_dir*."""
        from tess_pipeline.io.export import save_results

        save_results(self, Path(output_dir))

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary dict (no heavy objects)."""
        return {
            "target": self.target,
            "period": self.period,
            "detection": {
                k: v
                for k, v in self.detection.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            "stellar": {
                k: v
                for k, v in self.stellar.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            "planet": self.planet,
            "diagnostics": self.diagnostics,
            "metadata": {
                k: str(v)
                for k, v in self.metadata.items()
            },
        }
