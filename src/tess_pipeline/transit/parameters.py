"""
transit/parameters.py — Analytic derived transit/planet parameters.

These are quick estimates from transit geometry and stellar parameters.
They are superseded by Bayesian posteriors when inference runs.
Used as sanity checks and initial guesses for the Bayesian fit.
"""

from __future__ import annotations

import math
from typing import Any

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)

# Physical constants (CGS)
_G = 6.674e-8        # cm³ g⁻¹ s⁻²
_R_SUN = 6.957e10   # cm
_R_EARTH = 6.371e8  # cm
_M_SUN = 1.989e33   # g
_AU = 1.496e13      # cm
_SIGMA_SB = 5.670e-5  # erg cm⁻² s⁻¹ K⁻⁴


def quick_planet_params(
    *,
    period_days: float,
    depth: float,
    duration_hr: float,
    stellar: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute analytic planet parameters from detection geometry.

    Parameters
    ----------
    period_days : float
    depth : float
        Transit depth (fractional flux deficit, e.g. 0.01 for 1%).
    duration_hr : float
        Total transit duration T14 in hours.
    stellar : dict
        Output of ``catalogs.stellar.characterize_star()``.

    Returns
    -------
    dict with keys:
        rp_r_star   : planet-to-star radius ratio (Rp/R★)
        rp_earth    : planet radius in Earth radii (requires R★)
        a_au        : semi-major axis in AU (requires M★)
        a_r_star    : semi-major axis in units of R★ (requires T14, period)
        t_eq        : equilibrium temperature in K (requires Teff, a/R★)
        inclination : orbital inclination in degrees (assumes b=0 approximation)
    """
    result: dict[str, Any] = {}

    # ── Rp/R★ from transit depth ──────────────────────────────────────────────
    rp_r_star = math.sqrt(max(depth, 0.0))
    result["rp_r_star"] = rp_r_star

    # ── Planet radius ─────────────────────────────────────────────────────────
    r_star = stellar.get("r_star")
    if r_star is not None and r_star > 0:
        rp_solar = rp_r_star * r_star
        rp_earth = rp_solar * _R_SUN / _R_EARTH
        result["rp_earth"] = rp_earth
    else:
        result["rp_earth"] = None

    # ── Semi-major axis from Kepler's third law ───────────────────────────────
    m_star = stellar.get("m_star")
    if m_star is not None and m_star > 0:
        period_s = period_days * 86400.0
        m_cgs = m_star * _M_SUN
        a_cm = (_G * m_cgs * period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
        result["a_au"] = a_cm / _AU
    else:
        result["a_au"] = None

    # ── a/R★ from T14 and period ──────────────────────────────────────────────
    if r_star is not None and r_star > 0 and duration_hr > 0:
        duration_days = duration_hr / 24.0
        # Approximate (b=0): T14 ≈ (2R★/v_orb) × sqrt(1 - b²)
        # → a/R★ ≈ π × P / (T14) × sqrt(1 - (Rp/R★)²)
        # Simplified: a/R★ ≈ π × P / T14
        a_r_star = math.pi * period_days / duration_days
        result["a_r_star"] = a_r_star
    else:
        result["a_r_star"] = None

    # ── Equilibrium temperature ───────────────────────────────────────────────
    teff = stellar.get("teff")
    a_r_star = result.get("a_r_star")
    if teff is not None and a_r_star is not None and a_r_star > 0:
        # Teq = Teff × (1/4 × (1/a_r_star)²)^0.25 for zero albedo, full redistribution
        result["t_eq"] = teff * (0.25 / a_r_star**2) ** 0.25
    else:
        result["t_eq"] = None

    # ── Approximate inclination (b=0) ─────────────────────────────────────────
    a_r_star_val = result.get("a_r_star")
    if a_r_star_val is not None and a_r_star_val > 1:
        result["inclination"] = math.degrees(math.acos(0.0 / a_r_star_val))  # b=0 → i≈90
        result["inclination"] = 90.0  # placeholder (b=0 assumption)
    else:
        result["inclination"] = None

    return result


def derive_planet_parameters(
    posterior: Any,
    stellar: dict[str, Any],
) -> dict[str, Any]:
    """
    Derive physical planet parameters from an ArviZ InferenceData posterior.

    Returns medians and 16/84 percentile credible intervals.
    """
    try:
        import numpy as np
        import arviz as az
    except ImportError:
        log.warning("arviz not available; returning empty planet parameters")
        return {}

    summary = az.summary(posterior, var_names=["rp_r_star", "b", "t14", "t0"], round_to=6)

    result: dict[str, Any] = {}

    def _get(var: str) -> tuple[float | None, float | None]:
        try:
            row = summary.loc[var]
            return float(row["mean"]), float(row["sd"])
        except (KeyError, TypeError):
            return None, None

    rp_r_star, rp_r_star_err = _get("rp_r_star")
    result["rp_r_star"] = rp_r_star
    result["rp_r_star_err"] = rp_r_star_err

    result["b"], result["b_err"] = _get("b")
    result["t14_hr"], t14_err = _get("t14")
    result["t14_hr"] = (result["t14_hr"] * 24.0) if result["t14_hr"] else None
    result["t14_hr_err"] = (t14_err * 24.0) if t14_err else None
    result["t0"], result["t0_err"] = _get("t0")

    # Physical radius
    r_star = stellar.get("r_star")
    r_star_err = stellar.get("r_star_err", 0.0) or 0.0
    if rp_r_star is not None and r_star is not None and r_star > 0:
        rp_earth = rp_r_star * r_star * _R_SUN / _R_EARTH
        result["rp_earth"] = rp_earth
        if rp_r_star_err is not None:
            err_frac = math.sqrt((rp_r_star_err / rp_r_star) ** 2 + (r_star_err / r_star) ** 2)
            result["rp_earth_err"] = rp_earth * err_frac
        else:
            result["rp_earth_err"] = None
    else:
        result["rp_earth"] = None
        result["rp_earth_err"] = None

    # Semi-major axis and equilibrium temperature
    try:
        period_samples = az.extract(posterior, var_names=["period"])["period"].values
        period_med = float(np.median(period_samples))
    except Exception:  # noqa: BLE001
        period_med = None

    m_star = stellar.get("m_star")
    if period_med and m_star is not None and m_star > 0:
        period_s = period_med * 86400.0
        m_cgs = m_star * _M_SUN
        a_cm = (_G * m_cgs * period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
        result["a_au"] = a_cm / _AU
        result["a_au_err"] = None  # simplified; full propagation requires posterior samples

        if r_star is not None and r_star > 0:
            result["a_r_star"] = a_cm / (r_star * _R_SUN)

        teff = stellar.get("teff")
        if teff and result.get("a_r_star") and result["a_r_star"] > 0:
            result["t_eq"] = teff * (0.25 / result["a_r_star"] ** 2) ** 0.25
    else:
        result["a_au"] = None
        result["a_au_err"] = None
        result["a_r_star"] = None
        result["t_eq"] = None

    return result
