"""
transit/batman_model.py — Fast analytic transit model via batman.

Used for:
  * Diagnostic overlays on phase-folded light curves
  * MAP starting point for the Bayesian fit
  * Quick-look fits when [inference] is not installed

Reference: Mandel & Agol 2002; batman by Kreidberg 2015
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def make_batman_model(
    time: np.ndarray,
    *,
    t0: float,
    period: float,
    rp: float,
    a: float,
    inc: float = 90.0,
    ecc: float = 0.0,
    omega: float = 90.0,
    u: tuple[float, float] = (0.3, 0.2),
    limb_dark: str = "quadratic",
) -> np.ndarray:
    """
    Generate a batman transit light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array (same units as t0 and period, typically BTJD).
    t0 : float
        Transit mid-time.
    period : float
        Orbital period (days).
    rp : float
        Planet-to-star radius ratio Rp/R★.
    a : float
        Semi-major axis in units of R★.
    inc : float
        Orbital inclination (degrees).
    ecc : float
        Eccentricity.
    omega : float
        Argument of periastron (degrees).
    u : tuple[float, float]
        Quadratic limb-darkening coefficients (u1, u2).
    limb_dark : str
        Limb-darkening law (default: "quadratic").

    Returns
    -------
    np.ndarray
        Model flux array (same shape as *time*).
    """
    try:
        import batman
    except ImportError as exc:
        raise ImportError(
            "batman-package is required for analytic transit models. "
            "Install with: pip install batman-package"
        ) from exc

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = omega
    params.u = list(u)
    params.limb_dark = limb_dark

    m = batman.TransitModel(params, time)
    return m.light_curve(params)


def quick_batman_fit(
    lc: Any,
    period: float,
    stellar: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Fit a batman transit model to a phase-folded light curve via least squares.

    This is a simplified single-iteration MAP estimate, not a full posterior.
    Used when [inference] is not installed.

    Returns
    -------
    dict with keys:
        time, flux_model, residuals, planet_params (dict)
    """
    import numpy as np
    from scipy.optimize import minimize

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    flux_err = (
        np.asarray(lc.flux_err.value, dtype=float)
        if lc.flux_err is not None
        else np.ones_like(flux) * 1e-3
    )

    # ── Initial parameter guesses ─────────────────────────────────────────────
    r_star = (stellar or {}).get("r_star", 1.0) or 1.0
    depth = 1.0 - float(np.percentile(flux, 1))
    rp_init = math.sqrt(max(depth, 1e-6))
    t0_init = float(time[np.argmin(flux)])

    # Approximate a/R★
    a_init = 10.0  # default; overridden if stellar params available
    m_star = (stellar or {}).get("m_star")
    if m_star is not None:
        period_s = period * 86400.0
        G = 6.674e-8
        M_SUN = 1.989e33
        R_SUN = 6.957e10
        m_cgs = m_star * M_SUN
        a_cm = (G * m_cgs * period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
        a_init = a_cm / (r_star * R_SUN)

    def _chi2(theta: np.ndarray) -> float:
        t0, rp, a, u1 = theta
        try:
            model = make_batman_model(
                time,
                t0=t0,
                period=period,
                rp=max(rp, 1e-4),
                a=max(a, 1.1),
                u=(u1, 0.2),
            )
        except Exception:  # noqa: BLE001
            return 1e10
        return float(np.sum(((flux - model) / flux_err) ** 2))

    x0 = np.array([t0_init, rp_init, a_init, 0.3])
    bounds = [
        (t0_init - period * 0.05, t0_init + period * 0.05),
        (1e-4, 0.5),
        (1.1, 200.0),
        (0.0, 1.0),
    ]

    try:
        opt = minimize(_chi2, x0, method="L-BFGS-B", bounds=bounds)
        t0_fit, rp_fit, a_fit, u1_fit = opt.x
    except Exception as exc:  # noqa: BLE001
        log.warning("batman MAP fit failed: %s; using initial guess", exc)
        t0_fit, rp_fit, a_fit, u1_fit = t0_init, rp_init, a_init, 0.3

    flux_model = make_batman_model(
        time,
        t0=t0_fit,
        period=period,
        rp=rp_fit,
        a=a_fit,
        u=(u1_fit, 0.2),
    )

    R_EARTH_R_SUN = 6.371e8 / 6.957e10
    rp_earth = (rp_fit * r_star) / R_EARTH_R_SUN

    return {
        "time": time,
        "flux_model": flux_model,
        "residuals": flux - flux_model,
        "planet_params": {
            "t0": t0_fit,
            "rp_r_star": rp_fit,
            "a_r_star": a_fit,
            "rp_earth": rp_earth,
            "u1": u1_fit,
            "method": "batman_map",
        },
    }
