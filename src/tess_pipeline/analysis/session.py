"""
session.py — Step-by-step TESS exoplanet analysis session.

Use in notebooks by calling each stage method in order::

    from tess_pipeline import TESSAnalysis

    analysis = TESSAnalysis("TIC 307210830")
    analysis.resolve_target()
    analysis.lookup_archive_period()
    analysis.load_lightcurves()
    analysis.preprocess()
    analysis.search_period()
    analysis.query_gaia()
    analysis.characterize_star()
    analysis.query_sdss()
    analysis.fit_transit()
    analysis.derive_planet_parameters()
    analysis.check_convergence()
    analysis.generate_figures()

    analysis.results.summary()
"""

from __future__ import annotations

import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from tess_pipeline.analysis.stages import (
    InferenceStage,
    LightCurveStage,
    PeriodStage,
    StellarStage,
    TargetStage,
    VisualizationStage,
)
from tess_pipeline.config import PipelineConfig, build_config
from tess_pipeline.results import PipelineResults
from tess_pipeline.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

try:
    _PACKAGE_VERSION = version("tess-pipeline")
except PackageNotFoundError:
    _PACKAGE_VERSION = "0.1.0"


class TESSAnalysis:
    """
    Interactive TESS exoplanet analysis session.

    Each public method runs one pipeline stage and returns ``self`` so calls
    can be chained. Intermediate outputs are stored on :attr:`results`.

    Parameters
    ----------
    target : str | int | None
        TIC identifier for MAST download mode: ``"TIC 307210830"``, ``"TIC307210830"``,
        or ``307210830``. Optional when ``lightcurve_source="fits"``; TIC ID and
        coordinates are then read from FITS headers.
    inference : bool
        Run Bayesian MCMC fit (default ``True``). Set ``False`` for detection-only.
    period : float | None
        Override orbital period in days; skips archive lookup and TLS/BLS search.
    **kwargs
        Any :class:`~tess_pipeline.config.PipelineConfig` field accepted by
        :func:`~tess_pipeline.config.build_config`.
    """

    def __init__(
        self,
        target: str | int | None = None,
        *,
        inference: bool = True,
        period: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.config: PipelineConfig = build_config(
            target,
            inference=inference,
            period_override=period,
            **kwargs,
        )
        self.results = PipelineResults()
        self._init_metadata()

        self._target = TargetStage(self.config, self.results)
        self._lightcurve = LightCurveStage(self.config, self.results)
        self._period = PeriodStage(self.config, self.results)
        self._stellar = StellarStage(self.config, self.results)
        self._inference = InferenceStage(self.config, self.results)
        self._visualization = VisualizationStage(self.config, self.results)

    def _init_metadata(self) -> None:
        cfg = self.config
        self.results.metadata = {
            "tess_pipeline_version": _PACKAGE_VERSION,
            "run_start": datetime.datetime.utcnow().isoformat(),
            "config": {
                "target": cfg.target,
                "author": cfg.author,
                "cadence": cfg.cadence,
                "sectors": cfg.sectors,
                "lightcurve_source": cfg.lightcurve_source,
                "lightcurve_fits": [str(p) for p in cfg.lightcurve_fits],
                "search_method": cfg.search_method,
                "inference": cfg.inference,
                "inference_backend": cfg.inference_backend,
                "chains": cfg.chains,
                "draws": cfg.draws,
                "gp_kernel": cfg.gp_kernel,
                "verbose": cfg.verbose,
            },
        }

    # ── Target ───────────────────────────────────────────────────────────────

    def resolve_target(self) -> TESSAnalysis:
        """Resolve the target string to a TIC ID and sky coordinates."""
        configure_logging(debug=self.config.verbose)
        self._target.run()
        return self

    # ── Light curve ──────────────────────────────────────────────────────────

    def load_lightcurves(self) -> TESSAnalysis:
        """Download or load raw TESS light curves."""
        self._lightcurve.load()
        return self

    def preprocess(self) -> TESSAnalysis:
        """Sigma-clip, stitch, and flatten loaded light curves."""
        self._lightcurve.preprocess()
        if self.config.plots:
            from tess_pipeline.visualization import lightcurve as vlc
            if self.results.lightcurve is not None:
                fig_raw = vlc.plot_raw_lightcurve(self.results.lightcurve)
                self.results.figures["raw"] = fig_raw
                self._save_step_plot("raw", fig_raw)

                fig_flat = vlc.plot_flattened_lightcurve(self.results.lightcurve)
                self.results.figures["flat"] = fig_flat
                self._save_step_plot("flat", fig_flat)
        return self

    # ── Period ───────────────────────────────────────────────────────────────

    def lookup_archive_period(self) -> TESSAnalysis:
        """Query NASA Exoplanet Archive for a published period."""
        self._period.lookup_archive()
        return self

    def search_period(self) -> TESSAnalysis:
        """Run TLS/BLS period search (or diagnostics when archive period exists)."""
        self._period.search()
        if self.config.plots:
            from tess_pipeline.visualization import periodogram as vper
            from tess_pipeline.visualization import transit as vtr
            det = self.results.detection
            if det:
                if "tls_result" in det:
                    fig_per = vper.plot_tls_periodogram(det["tls_result"])
                    self.results.figures["tls_periodogram"] = fig_per
                    self._save_step_plot("tls_periodogram", fig_per)
                if "bls_result" in det:
                    fig_per = vper.plot_bls_periodogram(det["bls_result"])
                    self.results.figures["bls_periodogram"] = fig_per
                    self._save_step_plot("bls_periodogram", fig_per)
            
            if self.results.lightcurve is not None and self.results.period:
                fig_phase = vtr.plot_phase_curve(
                    self.results.lightcurve,
                    period=self.results.period["value"],
                    epoch=self.results.detection.get("epoch"),
                )
                self.results.figures["phase"] = fig_phase
                self._save_step_plot("phase", fig_phase)
        return self

    # ── Stellar ──────────────────────────────────────────────────────────────

    def query_gaia(self) -> TESSAnalysis:
        """Query Gaia DR3 for the target star."""
        self._stellar.query_gaia()
        return self

    def characterize_star(self) -> TESSAnalysis:
        """Derive stellar parameters (Teff, logg, R★, ρ★) from Gaia data."""
        self._stellar.characterize()
        return self

    def query_sdss(self) -> TESSAnalysis:
        """Query SDSS for radial velocity (optional; may return None)."""
        self._stellar.query_sdss()
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def fit_transit(self) -> TESSAnalysis:
        """Run Bayesian MCMC or batman analytic transit fit."""
        period = self._period.best_period
        self._inference.fit_transit(period=period)
        if self.config.plots and self.results.posterior is not None:
            from tess_pipeline.visualization import transit as vtr
            from tess_pipeline.visualization import posterior as vpost
            from tess_pipeline.visualization import diagnostics as vdiag

            fig_fit = vtr.plot_bayesian_fit(
                self.results.lightcurve, self.results.posterior, self.results.model
            )
            self.results.figures["bayesian_fit"] = fig_fit
            self._save_step_plot("bayesian_fit", fig_fit)

            fig_corner = vpost.plot_corner(self.results.posterior)
            self.results.figures["corner"] = fig_corner
            self._save_step_plot("corner", fig_corner)

            fig_trace = vpost.plot_trace(self.results.posterior)
            self.results.figures["trace"] = fig_trace
            self._save_step_plot("trace", fig_trace)

            fig_pred = vpost.plot_posterior_predictive(
                self.results.lightcurve, self.results.posterior, self.results.model
            )
            self.results.figures["posterior_predictive"] = fig_pred
            self._save_step_plot("posterior_predictive", fig_pred)

            if self.results.period:
                fig_res = vdiag.plot_residuals(
                    self.results.lightcurve,
                    self.results.model or {},
                    period=self.results.period["value"],
                    epoch=self.results.detection.get("epoch"),
                )
                self.results.figures["residuals"] = fig_res
                self._save_step_plot("residuals", fig_res)
        return self

    def derive_planet_parameters(self) -> TESSAnalysis:
        """Compute physical planet parameters from the posterior."""
        self._inference.derive_planet_parameters()
        return self

    def check_convergence(self) -> TESSAnalysis:
        """Check MCMC convergence (R-hat, ESS, divergences)."""
        self._inference.check_convergence()
        return self

    # ── Visualization & export ───────────────────────────────────────────────

    def _save_step_plot(self, name: str, fig: Any) -> None:
        """Save a progress plot for the given step if plots are enabled."""
        if not self.config.plots or fig is None:
            return
        from pathlib import Path
        tic = self.results.target.get("tic_id", "unknown")
        target_dir = Path(self.config.output_dir) / f"TIC {tic}"
        plots_dir = target_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            prefix_map = {
                "raw": "01_raw",
                "flat": "02_flat",
                "tls_periodogram": "03_tls_periodogram",
                "bls_periodogram": "03_bls_periodogram",
                "phase": "04_phase",
                "residuals": "05_residuals",
                "bayesian_fit": "06_bayesian_fit",
                "corner": "07_corner",
                "trace": "08_trace",
                "posterior_predictive": "09_posterior_predictive",
            }
            filename = prefix_map.get(name, name)
            if filename.startswith("phase_p"):
                filename = filename.replace("phase_p", "04_phase_p")
            path = plots_dir / f"{filename}.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            log.info("Saved step plot: %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save step plot %r: %s", name, exc)

    def generate_figures(self) -> TESSAnalysis:
        """Build diagnostic matplotlib figures."""
        if self.config.plots:
            self._visualization.generate()
        return self

    def save(self, output_dir: str | None = None) -> TESSAnalysis:
        """Write results to disk (CSV, JSON, figures, optional report)."""
        from pathlib import Path

        out = Path(output_dir) if output_dir is not None else self.config.output_dir
        self.results.save(out)
        return self

    # ── Full run ─────────────────────────────────────────────────────────────

    def run(self) -> PipelineResults:
        """Execute all stages in order and return :attr:`results`."""
        cfg = self.config
        configure_logging(debug=cfg.verbose)

        self.resolve_target()
        self.lookup_archive_period()
        self.load_lightcurves()
        self.preprocess()
        self.search_period()

        self.query_gaia()
        self.characterize_star()
        self.query_sdss()

        if cfg.inference:
            self.fit_transit()
            if cfg.inference_backend == "exoplanet":
                self.derive_planet_parameters()
                self.check_convergence()

        if cfg.plots:
            self.generate_figures()

        self.results.metadata["run_end"] = datetime.datetime.utcnow().isoformat()
        tic_id = self.results.target.get("tic_id", "?")
        log.info("Analysis complete for TIC %s", tic_id)
        return self.results
