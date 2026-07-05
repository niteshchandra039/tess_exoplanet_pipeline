"""
inference/bayesian.py — PyMC + exoplanet Bayesian transit fit.

Implements the primary parameter estimation workflow following:
  - TESS-Keck Survey XV (Polanski et al. 2023)
  - exoplanet TESS quick-fit tutorial
  - Fit T14 directly (avoids circular stellar density bias; Gilbert et al. 2022)
  - Kipping (2013) limb-darkening parameterization
  - GP noise model (celerite2 SHO)
  - NUTS via PyMC (4 chains, target_accept ≥ 0.9)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tess_pipeline.exceptions import InferenceNotInstalledError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def run_bayesian_fit(
    lc: Any,
    *,
    period: float,
    epoch: float | None,
    stellar: dict[str, Any],
    chains: int = 4,
    draws: int = 1500,
    tune: int = 1000,
    target_accept: float = 0.95,
    gp_kernel: str = "SHO",
    detections: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Build and sample a PyMC transit model (supports multiple planets).

    Parameters
    ----------
    lc : lightkurve.LightCurve
    period : float
        Best-fit or archived orbital period (days) for Planet 1.
    epoch : float | None
        Transit mid-time (BTJD) for Planet 1.
    stellar : dict
        Output of ``catalogs.stellar.characterize_star()``.
    chains, draws, tune : int
        PyMC NUTS settings.
    target_accept : float
        NUTS target acceptance rate.
    gp_kernel : str
        "SHO" or "Matern32".
    detections : list of dicts, optional
        A list of detection dictionaries, one for each planet to fit.

    Returns
    -------
    (arviz.InferenceData, dict)
        posterior InferenceData and model_outputs dict
        {'time', 'flux_model', 'gp_model', 'residuals', 'light_curve_p0', ...}
    """
    from tess_pipeline.inference.deps import check_inference_installed

    check_inference_installed()

    try:
        import pymc as pm
        import exoplanet as xo
        import pytensor.tensor as pt
        import arviz as az
    except ImportError as exc:
        raise InferenceNotInstalledError(
            "Bayesian inference packages failed to import after installation check. "
            "Reinstall the package with: pip install -e ."
        ) from exc

    # Clean duplicates and sort to satisfy celerite2's strict sorting requirement
    time_raw = np.asarray(lc.time.value, dtype=float)
    flux_raw = np.asarray(lc.flux.value, dtype=float)
    flux_err_raw = (
        np.asarray(lc.flux_err.value, dtype=float)
        if lc.flux_err is not None
        else np.ones_like(flux_raw) * 1e-3
    )

    sidx = np.argsort(time_raw)
    time_sorted = time_raw[sidx]
    flux_sorted = flux_raw[sidx]
    flux_err_sorted = flux_err_raw[sidx]

    uniq_mask = np.concatenate([[True], np.diff(time_sorted) > 0])
    time = time_sorted[uniq_mask]
    flux = flux_sorted[uniq_mask]
    flux_err = flux_err_sorted[uniq_mask]

    # Estimate transit depth from minimum flux
    depth_estimate = float(1.0 - np.percentile(flux, 1))

    if not detections:
        detections = [{"period": period, "epoch": epoch, "depth": depth_estimate}]

    periods = np.array([det["period"] for det in detections])
    epochs = np.array([det["epoch"] if det["epoch"] is not None else float(time[np.argmin(flux)]) for det in detections])
    depths = np.array([det.get("depth", depth_estimate) for det in detections])
    n_planets = len(detections)

    log.info("Building PyMC model (N=%d points, %d planets)", len(time), n_planets)

    from tess_pipeline.inference.priors import (
        add_stellar_priors,
        add_orbital_priors,
        add_transit_shape_priors,
        add_systematic_priors,
    )
    from tess_pipeline.inference.gp import build_gp

    with pm.Model() as model:

        # ── Priors ────────────────────────────────────────────────────────────
        stellar_vars = add_stellar_priors(model, stellar)
        orbital_vars = add_orbital_priors(model, period=periods, epoch=epochs, period_fixed=False)
        shape_vars = add_transit_shape_priors(model, depth_estimate=depths, fit_duration=False, n_planets=n_planets)
        sys_vars = add_systematic_priors(model, n_sectors=1)

        rho_star_var = stellar_vars["rho_star"]
        t0 = orbital_vars["t0"]
        period_var = orbital_vars["period"]
        rp_r_star = shape_vars["rp_r_star"]
        b = shape_vars["b"]
        u1 = shape_vars["u1"]
        u2 = shape_vars["u2"]
        mean_flux = sys_vars["mean_flux"]

        # ── Keplerian orbit ───────────────────────────────────────────────────
        orbit = xo.orbits.KeplerianOrbit(
            period=period_var,
            t0=t0,
            b=b,
            rho_star=rho_star_var,
            ror=rp_r_star,
        )

        # ── Transit light curve ───────────────────────────────────────────────
        star = xo.LimbDarkLightCurve(u1, u2)
        light_curves = star.get_light_curve(
            orbit=orbit,
            r=rp_r_star,
            t=time,
        )

        # Save individual planet light curves
        for idx in range(n_planets):
            pm.Deterministic(f"light_curve_p{idx}", light_curves[:, idx])

        transit_model = pm.Deterministic("transit_model", pm.math.sum(light_curves, axis=-1) + mean_flux[0] + 1.0)

        # ── GP noise model ────────────────────────────────────────────────────
        gp = build_gp(
            model,
            time=time,
            flux_err=flux_err,
            mean=transit_model,
            kernel=gp_kernel,
        )

        # ── Likelihood ────────────────────────────────────────────────────────
        gp.marginal("obs", observed=flux)

        # ── GP prediction ─────────────────────────────────────────────────────
        gp_pred = pm.Deterministic("gp_pred", gp.predict(flux, include_mean=False))

        # ── Full model ────────────────────────────────────────────────────────
        flux_model = pm.Deterministic("flux_model", transit_model + gp_pred)

        # ── Dense phase grid model for plotting uncertainty bands ──────────────
        phase_grid = np.linspace(-0.3, 0.3, 200)

        for idx in range(n_planets):
            # Create a single-planet orbit for this planet
            single_orbit = xo.orbits.KeplerianOrbit(
                period=period_var[idx],
                t0=t0[idx],
                b=b[idx],
                rho_star=rho_star_var,
                ror=rp_r_star[idx],
            )
            lc_pred_curve = star.get_light_curve(
                orbit=single_orbit,
                r=rp_r_star[idx],
                t=t0[idx] + phase_grid,
            )
            pm.Deterministic(f"lc_pred_p{idx}", lc_pred_curve[:, 0] + mean_flux[0] + 1.0)

        # ── NUTS initialization and sampling ──────────────────────────────────
        rp_inits = np.array([math.sqrt(max(d, 1e-5)) for d in depths])
        init_dict = {
            "period": periods,
            "t0": epochs,
            "b": np.array([0.1] * n_planets),
            "q1": 0.3,
            "q2": 0.3,
            "mean_flux": np.array([0.0]),
            "log_jitter": -6.0,
            "log_sigma_gp": -3.0,
            "log_rho_gp": np.log(10.0),
            "log_rp": np.log(rp_inits),
        }
        if "rho_star" in stellar and stellar["rho_star"] is not None:
            init_dict["rho_star"] = stellar["rho_star"]
        else:
            init_dict["rho_star"] = 1.40984

        log.info(
            "Sampling: %d chains × %d draws (tune=%d, target_accept=%.2f)",
            chains, draws, tune, target_accept,
        )
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 8),
            target_accept=target_accept,
            initvals=init_dict,
            init="adapt_diag",
            return_inferencedata=True,
            progressbar=True,
        )
        pm.compute_log_likelihood(trace)

    log.info("Sampling complete")

    # ── Extract model outputs ─────────────────────────────────────────────────
    model_outputs = _extract_model_outputs(trace, time, flux)

    return trace, model_outputs


def _extract_model_outputs(
    trace: Any,
    time: np.ndarray,
    flux: np.ndarray,
) -> dict[str, Any]:
    """Extract median model curve and residuals from the posterior (supports multiple planets)."""
    try:
        import numpy as np

        outputs = {
            "time": time,
            "flux_model": None,
            "gp_model": None,
            "transit_model": None,
            "residuals": None,
        }

        if hasattr(trace, "posterior") and "flux_model" in trace.posterior:
            outputs["flux_model"] = np.median(trace.posterior["flux_model"].values, axis=(0, 1))
            outputs["gp_model"] = np.median(trace.posterior["gp_pred"].values, axis=(0, 1)) if "gp_pred" in trace.posterior else None
            outputs["transit_model"] = np.median(trace.posterior["transit_model"].values, axis=(0, 1)) if "transit_model" in trace.posterior else None
            outputs["residuals"] = flux - outputs["flux_model"]

            # Extract individual planet models
            i = 0
            while True:
                key = f"light_curve_p{i}"
                if key in trace.posterior:
                    outputs[key] = np.median(trace.posterior[key].values, axis=(0, 1))
                    i += 1
                else:
                    break

        return outputs
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not extract model outputs: %s", exc)
        return {
            "time": time,
            "flux_model": None,
            "gp_model": None,
            "transit_model": None,
            "residuals": None,
        }


def calculate_bic(trace: Any, n_planets: int) -> float:
    """
    Calculate the Bayesian Information Criterion (BIC) from a PyMC trace.
    
    BIC = k * ln(N) - 2 * ln(L_max)
    """
    import numpy as np

    if not hasattr(trace, "log_likelihood") or "obs" not in trace.log_likelihood:
        raise ValueError("Trace does not contain log-likelihood group 'obs'")

    # Extract log-likelihood array of shape (chains, draws, N_points)
    log_lik_array = trace.log_likelihood["obs"].values
    
    # Sum log-likelihood over all data points for each draw
    sum_log_lik = np.sum(log_lik_array, axis=-1)  # shape: (chains, draws)
    
    # Find the maximum log-likelihood across all MCMC samples
    max_log_lik = float(np.max(sum_log_lik))
    
    # Number of data points
    n_points = int(log_lik_array.shape[-1])
    
    # Number of free parameters:
    # 7 base stellar/GP/mean parameters + 4 parameters per planet (period, t0, b, rp)
    k = 7 + 4 * n_planets
    
    bic = k * np.log(n_points) - 2.0 * max_log_lik
    return bic


