"""Bayesian transit fitting and derived planet parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.config import PipelineConfig
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


class InferenceStage:
    """Run MCMC transit fits or fast batman-only models."""

    def __init__(self, config: PipelineConfig, results: PipelineResults) -> None:
        self.config = config
        self.results = results

    def fit_transit(self, *, period: float) -> Any:
        """Run the configured inference backend."""
        cfg = self.config
        lc = self.results.lightcurve
        stellar = self.results.stellar
        if lc is None:
            raise RuntimeError("Light curve required; run preprocess() first")

        if cfg.inference and cfg.inference_backend == "exoplanet":
            from tess_pipeline.inference.deps import check_inference_installed

            check_inference_installed()

            log.info("Running Bayesian transit fit (exoplanet + PyMC)")
            from tess_pipeline.inference.bayesian import run_bayesian_fit

            posterior, model_outputs = run_bayesian_fit(
                lc,
                period=period,
                epoch=self.results.detection.get("epoch"),
                stellar=stellar,
                chains=cfg.chains,
                draws=cfg.draws,
                tune=cfg.tune,
                target_accept=cfg.target_accept,
                gp_kernel=cfg.gp_kernel,
            )
            self.results.posterior = posterior
            self.results.model = model_outputs
            return posterior

        if cfg.inference and cfg.inference_backend == "batman_only":
            log.info("Using batman analytic fit (no MCMC)")
            from tess_pipeline.transit.batman_model import quick_batman_fit

            batman_result = quick_batman_fit(lc, period, stellar)
            self.results.model = batman_result
            self.results.planet = batman_result.get("planet_params", {})
            return batman_result

        log.info("Inference disabled; skipping transit fit")
        return None

    def derive_planet_parameters(self) -> dict[str, Any]:
        """Compute physical planet parameters from the posterior."""
        if self.results.posterior is None:
            raise RuntimeError("Call fit_transit() before derive_planet_parameters()")

        from tess_pipeline.transit.parameters import derive_planet_parameters

        planet = derive_planet_parameters(self.results.posterior, self.results.stellar)
        self.results.planet = planet
        return planet

    def check_convergence(self) -> dict[str, Any]:
        """Run MCMC convergence diagnostics."""
        if self.results.posterior is None:
            raise RuntimeError("Call fit_transit() before check_convergence()")

        from tess_pipeline.inference.diagnostics import check_convergence

        diag = check_convergence(self.results.posterior)
        self.results.diagnostics = diag
        return diag
