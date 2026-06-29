"""
inference/priors.py — Centralized prior definitions for the PyMC transit model.

All priors are documented. Parameters:
  * Stellar: ρ★ Gaussian from isoclassify/gaia_only
  * Orbital: log-uniform period, Gaussian t0
  * Transit: Kipping (2013) q1/q2 limb darkening, ImpactParameter distribution for b
  * Systematic: per-sector flux offsets, jitter (InverseGamma)

Reference: Kipping 2013, MNRAS 435, 2152
           https://doi.org/10.1093/mnras/stt1435
"""

from __future__ import annotations

import math
from typing import Any


def add_stellar_priors(
    model: Any,
    stellar: dict[str, Any],
) -> dict[str, Any]:
    """
    Add stellar density prior to a PyMC model.

    Returns dict of PyMC variables: {'rho_star': ...}
    """
    import pymc as pm
    import numpy as np

    rho_star = stellar.get("rho_star")
    rho_star_err = stellar.get("rho_star_err")

    with model:
        if rho_star is not None and rho_star > 0:
            rho_err = rho_star_err if rho_star_err is not None else rho_star * 0.1
            rho_star_var = pm.TruncatedNormal(
                "rho_star",
                mu=rho_star,
                sigma=rho_err,
                lower=0.0,
                initval=rho_star,
            )
        else:
            # Broad log-normal when stellar density is unknown
            rho_star_var = pm.LogNormal(
                "rho_star",
                mu=np.log(1.4),   # solar density in g/cm³
                sigma=1.0,
                initval=1.4,
            )

    return {"rho_star": rho_star_var}


def add_orbital_priors(
    model: Any,
    *,
    period: float,
    epoch: float | None,
    period_fixed: bool = True,
) -> dict[str, Any]:
    """
    Add orbital parameter priors to a PyMC model.

    Parameters
    ----------
    period : float
        Best period from detection (days). Used as Gaussian mean.
    epoch : float | None
        Best epoch from detection (BTJD). Used as Gaussian mean.
    period_fixed : bool
        If True, fix period to the detection value (faster convergence).

    Returns dict of PyMC variables.
    """
    import pymc as pm
    import numpy as np

    with model:
        if period_fixed:
            period_var = pm.Data("period", period)
        else:
            period_var = pm.Normal(
                "period",
                mu=period,
                sigma=period * 0.01,   # 1% width
                initval=period,
            )

        if epoch is not None:
            t0_var = pm.Normal(
                "t0",
                mu=epoch,
                sigma=0.1,   # 0.1 day width on transit midtime
                initval=epoch,
            )
        else:
            t0_var = pm.Uniform(
                "t0",
                lower=epoch - period if epoch else 0.0,
                upper=epoch + period if epoch else period,
                initval=epoch,
            )

    return {"period": period_var, "t0": t0_var}


def add_transit_shape_priors(
    model: Any,
    *,
    depth_estimate: float | None = None,
) -> dict[str, Any]:
    """
    Add transit shape priors (Rp/R★, b, T14, limb darkening).

    Uses Kipping (2013) q1/q2 parameterization for quadratic limb darkening.
    """
    import pymc as pm
    import numpy as np

    rp_init = math.sqrt(depth_estimate) if depth_estimate else 0.1

    with model:
        # ── Planet-to-star radius ratio ────────────────────────────────────────
        log_rp = pm.Uniform(
            "log_rp",
            lower=np.log(0.001),
            upper=np.log(0.5),
            initval=np.log(rp_init),
        )
        rp_r_star = pm.Deterministic("rp_r_star", pm.math.exp(log_rp))

        # ── Impact parameter ──────────────────────────────────────────────────
        b = pm.Uniform("b", lower=0.0, upper=1.0 + rp_r_star, initval=0.1)

        # ── Transit duration T14 (days) ────────────────────────────────────────
        # Fit T14 directly (TESS-Keck XV approach; avoids circular density bias)
        log_t14 = pm.Uniform(
            "log_t14",
            lower=np.log(0.01),
            upper=np.log(0.5),
            initval=np.log(0.1),
        )
        t14 = pm.Deterministic("t14", pm.math.exp(log_t14))

        # ── Kipping (2013) limb darkening ──────────────────────────────────────
        q1 = pm.Uniform("q1", lower=0.0, upper=1.0, initval=0.3)
        q2 = pm.Uniform("q2", lower=0.0, upper=1.0, initval=0.3)

        # Transform to physical (u1, u2):  u1 = 2√q1·q2, u2 = √q1(1 - 2q2)
        sqrt_q1 = pm.math.sqrt(q1)
        u1 = pm.Deterministic("u1", 2.0 * sqrt_q1 * q2)
        u2 = pm.Deterministic("u2", sqrt_q1 * (1.0 - 2.0 * q2))

    return {
        "rp_r_star": rp_r_star,
        "b": b,
        "t14": t14,
        "q1": q1,
        "q2": q2,
        "u1": u1,
        "u2": u2,
    }


def add_systematic_priors(
    model: Any,
    *,
    n_sectors: int = 1,
) -> dict[str, Any]:
    """
    Add per-sector flux offset and white-noise jitter priors.
    """
    import pymc as pm

    with model:
        mean_flux = pm.Normal(
            "mean_flux",
            mu=0.0,
            sigma=0.01,
            shape=n_sectors,
            initval=[0.0] * n_sectors,
        )
        log_jitter = pm.Normal(
            "log_jitter",
            mu=-6.0,
            sigma=2.0,
            initval=-6.0,
        )
        jitter = pm.Deterministic("jitter", pm.math.exp(log_jitter))

    return {"mean_flux": mean_flux, "jitter": jitter}
