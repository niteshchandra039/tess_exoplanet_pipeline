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
            set_publication_style,
        )
        set_publication_style()

        cfg = self.config
        results = self.results
        figures: dict[str, Any] = {}

        log.info("Generating diagnostic figures")

        # Extract target metadata for titles
        tic_id = results.target.get("tic_id", "unknown")
        secs = results.metadata.get("sectors_used", [])
        if not secs and hasattr(cfg, "sectors"):
            secs = cfg.sectors
        if isinstance(secs, str):
            sectors_str = secs
        else:
            sectors_str = ", ".join(map(str, secs)) if secs else "all"

        if results.lightcurve is not None:
            figures["raw"] = vlc.plot_raw_lightcurve(results.lightcurve, tic_id=tic_id, sectors_str=sectors_str)
            figures["flat"] = vlc.plot_flattened_lightcurve(results.lightcurve, tic_id=tic_id, sectors_str=sectors_str)

        if results.detection:
            detections = results.metadata.get("detections", [results.detection])
            if any(k in results.detection for k in ("tls_result", "tls_result_broad", "tls_periods", "tls_periods_broad")):
                figures["tls_periodogram"] = vper.plot_tls_periodogram(
                    detections,
                    tic_id=tic_id,
                    sectors_str=sectors_str,
                    mode="auto"
                )
                for idx, det in enumerate(detections):
                    figures[f"tls_periodogram_coarse_p{idx}"] = vper.plot_tls_periodogram(
                        det,
                        tic_id=tic_id,
                        sectors_str=sectors_str,
                        mode="coarse"
                    )
                    figures[f"tls_periodogram_fine_p{idx}"] = vper.plot_tls_periodogram(
                        det,
                        tic_id=tic_id,
                        sectors_str=sectors_str,
                        mode="fine"
                    )
            if any(k in results.detection for k in ("bls_result", "bls_result_broad", "bls_periods", "bls_periods_broad")):
                figures["bls_periodogram"] = vper.plot_bls_periodogram(
                    detections,
                    tic_id=tic_id,
                    sectors_str=sectors_str,
                    mode="auto"
                )
                for idx, det in enumerate(detections):
                    figures[f"bls_periodogram_coarse_p{idx}"] = vper.plot_bls_periodogram(
                        det,
                        tic_id=tic_id,
                        sectors_str=sectors_str,
                        mode="coarse"
                    )
                    figures[f"bls_periodogram_fine_p{idx}"] = vper.plot_bls_periodogram(
                        det,
                        tic_id=tic_id,
                        sectors_str=sectors_str,
                        mode="fine"
                    )

        if results.lightcurve is not None and results.period:
            detections = results.metadata.get("detections", [])
            if len(detections) > 1:
                for idx, det in enumerate(detections):
                    figures[f"phase_p{idx}"] = vtr.plot_phase_curve(
                        results.lightcurve,
                        period=det["period"],
                        epoch=det["epoch"],
                        model=None,
                        tic_id=tic_id,
                        sectors_str=sectors_str
                    )
            else:
                figures["phase"] = vtr.plot_phase_curve(
                    results.lightcurve,
                    period=results.period["value"],
                    epoch=results.detection.get("epoch"),
                    model=results.model or None,
                    tic_id=tic_id,
                    sectors_str=sectors_str
                )
            figures["residuals"] = vdiag.plot_residuals(
                results.lightcurve,
                results.model or {},
                period=results.period["value"],
                epoch=results.detection.get("epoch"),
                tic_id=tic_id,
                sectors_str=sectors_str
            )

        if results.posterior is not None:
            figures["bayesian_fit"] = vtr.plot_bayesian_fit(
                results.lightcurve, results.posterior, results.model,
                tic_id=tic_id, sectors_str=sectors_str
            )
            figures["gp_acf"] = vdiag.plot_gp_acf(
                results.lightcurve, results.model or {},
                tic_id=tic_id, sectors_str=sectors_str
            )
            if results.planets:
                for idx, pl in enumerate(results.planets):
                    figures[f"transit_stack_p{idx}"] = vdiag.plot_transit_stack(
                        results.lightcurve,
                        period=pl["period"],
                        epoch=pl["t0"],
                        duration_hr=pl.get("t14_hr", 3.0),
                        tic_id=tic_id,
                        sectors_str=sectors_str,
                        gp_model=results.model.get("gp_model") if results.model else None
                    )
                    figures[f"mcmc_phase_p{idx}"] = vtr.plot_mcmc_phase_curve(
                        results.lightcurve,
                        results.posterior,
                        period=pl["period"],
                        epoch=pl["t0"],
                        planet_idx=idx,
                        tic_id=tic_id,
                        sectors_str=sectors_str
                    )
            else:
                p_val = results.period["value"]
                e_val = results.detection.get("epoch")
                dur_val = results.detection.get("duration_hr", 3.0)
                figures["transit_stack"] = vdiag.plot_transit_stack(
                    results.lightcurve,
                    period=p_val,
                    epoch=e_val,
                    duration_hr=dur_val,
                    tic_id=tic_id,
                    sectors_str=sectors_str,
                    gp_model=results.model.get("gp_model") if results.model else None
                )
                figures["mcmc_phase"] = vtr.plot_mcmc_phase_curve(
                    results.lightcurve,
                    results.posterior,
                    period=p_val,
                    epoch=e_val,
                    tic_id=tic_id,
                    sectors_str=sectors_str
                )
            figures["corner"] = vpost.plot_corner(results.posterior, tic_id=tic_id, sectors_str=sectors_str)
            figures["trace"] = vpost.plot_trace(results.posterior, tic_id=tic_id, sectors_str=sectors_str)
            figures["posterior_predictive"] = vpost.plot_posterior_predictive(
                results.lightcurve, results.posterior, results.model,
                tic_id=tic_id, sectors_str=sectors_str
            )

        results.figures = figures
        return figures
