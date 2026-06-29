"""Generate diagnostic figures from analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.config import PipelineConfig
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


class VisualizationStage:
    """Build matplotlib figures for each analysis stage."""

    def __init__(self, config: PipelineConfig, results: PipelineResults) -> None:
        self.config = config
        self.results = results

    def generate(self) -> dict[str, Any]:
        """Create all applicable diagnostic plots."""
        from tess_pipeline.visualization import (
            diagnostics as vdiag,
            lightcurve as vlc,
            periodogram as vper,
            posterior as vpost,
            transit as vtr,
        )

        cfg = self.config
        results = self.results
        figures: dict[str, Any] = {}

        log.info("Generating diagnostic figures")

        if results.lightcurve is not None:
            figures["raw"] = vlc.plot_raw_lightcurve(results.lightcurve)
            figures["flat"] = vlc.plot_flattened_lightcurve(results.lightcurve)

        if results.detection:
            if "tls_result" in results.detection:
                figures["tls_periodogram"] = vper.plot_tls_periodogram(
                    results.detection["tls_result"]
                )
            if "bls_result" in results.detection:
                figures["bls_periodogram"] = vper.plot_bls_periodogram(
                    results.detection["bls_result"]
                )

        if results.lightcurve is not None and results.period:
            figures["phase"] = vtr.plot_phase_curve(
                results.lightcurve,
                period=results.period["value"],
                epoch=results.detection.get("epoch"),
                model=results.model or None,
            )
            figures["residuals"] = vdiag.plot_residuals(
                results.lightcurve,
                results.model or {},
                period=results.period["value"],
                epoch=results.detection.get("epoch"),
            )

        if results.posterior is not None:
            figures["bayesian_fit"] = vtr.plot_bayesian_fit(
                results.lightcurve, results.posterior, results.model
            )
            figures["phase"] = vtr.plot_mcmc_phase_curve(
                results.lightcurve,
                results.posterior,
                period=results.period["value"],
                epoch=results.detection.get("epoch"),
            )
            figures["corner"] = vpost.plot_corner(results.posterior)
            figures["trace"] = vpost.plot_trace(results.posterior)
            figures["posterior_predictive"] = vpost.plot_posterior_predictive(
                results.lightcurve, results.posterior, results.model
            )

        results.figures = figures
        return figures
