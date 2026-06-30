"""
inference/gp.py — Gaussian Process noise model for residual stellar variability.

Default: SHO (Stochastically-driven Harmonic Oscillator) kernel via celerite2.
Alternative: Matérn ν=3/2 kernel (Saha et al. 2024).

The GP is marginalized jointly with transit parameters in the PyMC model.
"""

from __future__ import annotations

from typing import Any

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def build_gp(
    model: Any,
    *,
    time: Any,
    residuals: Any = None,
    flux_err: Any,
    mean: Any = 0.0,
    kernel: str = "SHO",
) -> Any:
    """
    Add a GP noise model to a PyMC model.

    Parameters
    ----------
    model : pymc.Model
    time : pytensor / numpy array
        Observation times.
    residuals : pytensor expression
        Transit-subtracted flux (data − transit model).
    flux_err : array
        Per-point flux uncertainties.
    kernel : str
        "SHO" (default) or "Matern32".

    Returns
    -------
    gp : celerite2.pymc.GaussianProcess
    """
    try:
        import celerite2.pymc as celerite2_pm
        from celerite2.pymc import terms as c2terms
        import pymc as pm
        import numpy as np
        import pytensor.tensor as pt
    except ImportError as exc:
        raise ImportError(
            "celerite2 and pymc are required for the GP noise model. "
            "Reinstall the package with: pip install -e ."
        ) from exc

    with model:
        # ── GP hyperparameters ────────────────────────────────────────────────
        log_sigma_gp = pm.Normal("log_sigma_gp", mu=-3.0, sigma=2.0, initval=-3.0)
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(10.0), sigma=2.0, initval=np.log(10.0))

        sigma_gp = pm.Deterministic("sigma_gp", pm.math.exp(log_sigma_gp))
        rho_gp = pm.Deterministic("rho_gp", pm.math.exp(log_rho_gp))

        if kernel == "SHO":
            # SHO: Q fixed at 1/sqrt(2) for a critically-damped oscillator
            term = c2terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1.0 / np.sqrt(2.0))
            log.debug("Using SHO GP kernel")
        elif kernel == "Matern32":
            term = c2terms.Matern32Term(sigma=sigma_gp, rho=rho_gp)
            log.debug("Using Matérn 3/2 GP kernel")
        else:
            raise ValueError(f"Unknown GP kernel: {kernel!r}. Use 'SHO' or 'Matern32'.")

        gp = celerite2_pm.GaussianProcess(term, mean=mean)
        gp.compute(time, diag=flux_err**2 + pm.math.exp(2.0 * model["log_jitter"]), quiet=True)

    return gp
