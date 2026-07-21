"""
inference/gp.py — Gaussian Process noise model for residual stellar variability.

Default: SHO (Stochastically-driven Harmonic Oscillator) kernel via celerite2.
Alternative: Matérn ν=3/2 kernel (Saha et al. 2024).

The GP is marginalized jointly with transit parameters in the PyMC model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def sigma_clip_light_curve(
    time: Any,
    flux: Any,
    flux_err: Any,
    *,
    sigma_upper: float = 6.0,
    sigma_lower: float = 20.0,
    max_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask extreme light-curve outliers using iterative robust clipping.

    The clipping is deliberately asymmetric. Positive excursions are usually
    flares, cosmic rays, or instrumental events and can be rejected at a
    moderately conservative threshold. Negative excursions are clipped only
    at a much larger threshold because planetary transits are real, negative
    signals. The returned arrays contain only retained finite observations;
    the boolean mask is aligned with the input arrays and is ``True`` for
    retained points.

    A robust median/MAD scale is combined with the reported per-point
    uncertainty. This prevents a single bad uncertainty estimate from making
    the clipping threshold arbitrarily small or large. The center and scale
    are recomputed after every iteration, so multiple adjacent outliers do
    not permanently bias the estimate.

    Parameters
    ----------
    time, flux, flux_err : array-like
        One-dimensional, equally sized observations. ``time`` is used for
        validation and returned unchanged for retained observations.
    sigma_upper : float, default=6
        Positive standardized-residual threshold.
    sigma_lower : float, default=20
        Absolute threshold for negative standardized residuals. The high
        default is intentional: it protects transit-shaped signals.
    max_iters : int, default=5
        Maximum number of clipping passes.

    Returns
    -------
    time_clean, flux_clean, flux_err_clean, mask
        Cleaned arrays and an input-aligned boolean retention mask.

    Raises
    ------
    ValueError
        If inputs are not one-dimensional, have different lengths, contain
        no valid observations, or contain invalid clipping parameters.
    """
    time_array = np.asarray(time, dtype=float)
    flux_array = np.asarray(flux, dtype=float)
    err_array = np.asarray(flux_err, dtype=float)

    if time_array.ndim != 1 or flux_array.ndim != 1 or err_array.ndim != 1:
        raise ValueError("time, flux, and flux_err must be one-dimensional")
    if not (len(time_array) == len(flux_array) == len(err_array)):
        raise ValueError("time, flux, and flux_err must have the same length")
    if sigma_upper <= 0 or sigma_lower <= 0 or max_iters < 1:
        raise ValueError("clipping thresholds must be positive and max_iters >= 1")

    # Do not let NaNs or non-positive uncertainties enter either the robust
    # estimator or the GP likelihood. They cannot provide useful information.
    mask = np.isfinite(time_array) & np.isfinite(flux_array) & np.isfinite(err_array) & (err_array > 0)
    if not np.any(mask):
        raise ValueError("no finite observations with positive flux errors")

    for _ in range(max_iters):
        values = flux_array[mask]
        errors = err_array[mask]
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        robust_sigma = 1.4826 * mad

        # The MAD can be zero for quantized or very quiet data. The error
        # median supplies a stable noise floor in that case.
        scale = max(robust_sigma, float(np.median(errors)), np.finfo(float).eps)
        standardized = (flux_array - center) / np.maximum(err_array, scale)
        new_mask = mask & (standardized <= sigma_upper) & (standardized >= -sigma_lower)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if not np.any(mask):
        raise ValueError("sigma clipping rejected every observation")
    return time_array[mask], flux_array[mask], err_array[mask], mask


# Descriptive alias for callers that prefer the operation-oriented name.
iterative_sigma_clip = sigma_clip_light_curve


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
    residuals : pytensor expression, optional
        Deprecated compatibility argument. The residuals must be marginalized
        through ``gp.marginal(..., observed=flux)`` with the transit model as
        the GP mean; this function therefore intentionally does not use this
        argument.
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
    except ImportError as exc:
        raise ImportError(
            f"celerite2 and pymc are required for the GP noise model (detailed error: {exc}). "
            "Reinstall the package with: pip install -e ."
        ) from exc

    with model:
        # ── GP hyperparameters ────────────────────────────────────────────────
        # Typical normalized TESS out-of-transit variability is at the
        # per-mille level. A truncated prior prevents the GP from explaining
        # percent-level transits or isolated artifacts as stellar variability.
        log_sigma_gp = pm.TruncatedNormal(
            "log_sigma_gp",
            mu=np.log(1.0e-3),
            sigma=0.5,
            lower=np.log(1.0e-5),
            upper=np.log(2.0e-2),
        )
        sigma_gp = pm.Deterministic("sigma_gp", pm.math.exp(log_sigma_gp))

        # Hard lower bound of two days removes the sub-day flexibility that
        # allowed the previous SHO prior to absorb transit ingress/egress.
        # The upper bound also avoids an effectively unconstrained constant
        # offset over a short TESS sector.
        log_rho_gp = pm.TruncatedNormal(
            "log_rho_gp",
            mu=np.log(10.0),
            sigma=0.35,
            lower=np.log(2.0),
            upper=np.log(100.0),
        )
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
