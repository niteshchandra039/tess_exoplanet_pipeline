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

            detections = self.results.metadata.get("detections", [])
            
            # If we have multiple planets, first fit the 1-planet model to compute BIC comparison
            bic1 = None
            if len(detections) >= 2:
                log.info("Multiple planets detected (%d). Running 1-planet model fit for comparison...", len(detections))
                try:
                    trace1, _ = run_bayesian_fit(
                        lc,
                        period=period,
                        epoch=self.results.detection.get("epoch"),
                        stellar=stellar,
                        chains=cfg.chains,
                        draws=cfg.draws,
                        tune=cfg.tune,
                        target_accept=cfg.target_accept,
                        gp_kernel=cfg.gp_kernel,
                        detections=detections[:1],  # fit only first planet
                    )
                    from tess_pipeline.inference.bayesian import calculate_bic
                    bic1 = calculate_bic(trace1, n_planets=1)
                    log.info("1-planet model BIC: %.2f", bic1)
                except Exception as exc:
                    log.warning("Could not fit 1-planet model for BIC comparison: %s", exc)
                    bic1 = None

            # Fit the full multi-planet model
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
                detections=detections,
            )
            self.results.posterior = posterior
            self.results.model = model_outputs

            # Compute model comparison statistics if we have multiple planets
            if len(detections) >= 2:
                if bic1 is not None:
                    try:
                        from tess_pipeline.inference.bayesian import calculate_bic
                        import numpy as np

                        bic_multi = calculate_bic(posterior, n_planets=len(detections))
                        delta_bic = bic_multi - bic1
                        prob_multi = float(1.0 / (1.0 + np.exp(delta_bic / 2.0)))
                        
                        bic_comp = {
                            "bic_1planet": bic1,
                            "bic_multiplanet": bic_multi,
                            "delta_bic": delta_bic,
                            "probability_multiplanet": prob_multi,
                        }
                        self.results.metadata["model_comparison"] = bic_comp
                        log.info(
                            "Model comparison: BIC(1p)=%.2f, BIC(%dp)=%.2f, delta=%.2f, prob=%.2f%%",
                            bic1, len(detections), bic_multi, delta_bic, prob_multi * 100.0
                        )
                        
                        # Update detections metadata with probabilities
                        for idx, det in enumerate(self.results.metadata["detections"]):
                            if idx == 0:
                                det["existence_probability_bayesian"] = 1.0
                                det["bayesian_confidence"] = "Confirmed"
                            else:
                                det["existence_probability_bayesian"] = prob_multi
                                if delta_bic < -10:
                                    det["bayesian_confidence"] = "Very Strong"
                                elif delta_bic < -6:
                                    det["bayesian_confidence"] = "Strong"
                                elif delta_bic < -2:
                                    det["bayesian_confidence"] = "Positive"
                                elif delta_bic < 0:
                                    det["bayesian_confidence"] = "Weak"
                                else:
                                    det["bayesian_confidence"] = "Not Justified"
                    except Exception as exc:
                        log.warning("Could not calculate model comparison: %s", exc)
            else:
                # 1 planet defaults
                if len(self.results.metadata.get("detections", [])) == 1:
                    det = self.results.metadata["detections"][0]
                    det["existence_probability_bayesian"] = 1.0
                    det["bayesian_confidence"] = "Confirmed"

            return posterior

        if cfg.inference and cfg.inference_backend == "batman_only":
            log.info("Using batman analytic fit (no MCMC)")
            from tess_pipeline.transit.batman_model import quick_batman_fit

            batman_result = quick_batman_fit(lc, period, stellar)
            self.results.model = batman_result
            self.results.planet = batman_result.get("planet_params", {})
            return batman_result

        if cfg.inference and cfg.inference_backend == "phoebe":
            log.info("Using PHOEBE 2 modeling backend")
            from tess_pipeline.transit.phoebe_model import run_phoebe_fit
            import numpy as np

            rv_times = np.asarray(cfg.rv_times) if cfg.rv_times else None
            rv_vals = np.asarray(cfg.rv_vals) if cfg.rv_vals else None
            rv_errs = np.asarray(cfg.rv_errs) if cfg.rv_errs else None

            phoebe_result = run_phoebe_fit(
                lc,
                period=period,
                stellar=stellar,
                rv_times=rv_times,
                rv_vals=rv_vals,
                rv_errs=rv_errs,
                input_is_magnitude=cfg.input_is_magnitude,
            )
            self.results.model = phoebe_result
            self.results.planet = phoebe_result.get("planet_params", {})
            return phoebe_result

        log.info("Inference disabled; skipping transit fit")
        return None

    def derive_planet_parameters(self) -> dict[str, Any]:
        """Compute physical planet parameters from the posterior."""
        if self.results.posterior is None:
            raise RuntimeError("Call fit_transit() before derive_planet_parameters()")

        from tess_pipeline.transit.parameters import derive_planet_parameters

        planets = derive_planet_parameters(self.results.posterior, self.results.stellar)
        if isinstance(planets, list):
            self.results.planets = planets
            if planets:
                self.results.planet = planets[0]
            else:
                self.results.planet = {}
        else:
            self.results.planet = planets
            self.results.planets = [planets]
        return self.results.planet

    def check_convergence(self) -> dict[str, Any]:
        """Run MCMC convergence diagnostics."""
        if self.results.posterior is None:
            raise RuntimeError("Call fit_transit() before check_convergence()")

        from tess_pipeline.inference.diagnostics import check_convergence

        diag = check_convergence(self.results.posterior)
        self.results.diagnostics = diag
        return diag
