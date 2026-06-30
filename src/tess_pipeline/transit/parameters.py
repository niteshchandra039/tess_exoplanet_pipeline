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
) -> list[dict[str, Any]]:
    """
    Derive physical planet parameters from an ArviZ InferenceData posterior (supports multiple planets).

    Returns list of dicts with medians, standard deviations, and 16th/84th percentiles.
    """
    try:
        import numpy as np
        import arviz as az
    except ImportError:
        log.warning("arviz not available; returning empty planet parameters")
        return []

    try:
        post = posterior.posterior
        if "rp_r_star" in post:
            val = post["rp_r_star"].values
            if val.ndim == 3:
                n_planets = val.shape[-1]
            else:
                n_planets = 1
        else:
            return []
    except (KeyError, AttributeError) as exc:
        log.warning("Required MCMC trace variables missing: %s", exc)
        return []

    planets_list = []

    # Get stellar parameters with uncertainties to propagate
    r_star = stellar.get("r_star", 1.0)
    r_star_err = stellar.get("r_star_err", 0.0) or 0.0
    teff = stellar.get("teff", 5777.0)
    teff_err = stellar.get("teff_err", 100.0) or 100.0

    for i in range(n_planets):
        try:
            if n_planets > 1:
                rp_r_star_samples = post["rp_r_star"].values[..., i].flatten()
                b_samples = post["b"].values[..., i].flatten()
                period_samples = post["period"].values[..., i].flatten()
                t0_samples = post["t0"].values[..., i].flatten()
            else:
                rp_r_star_samples = post["rp_r_star"].values.flatten()
                b_samples = post["b"].values.flatten()
                period_samples = post["period"].values.flatten()
                t0_samples = post["t0"].values.flatten()

            rho_star_samples = post["rho_star"].values.flatten()
        except KeyError as exc:
            log.warning("Missing planet %d MCMC trace variables: %s", i + 1, exc)
            continue

        result: dict[str, Any] = {}

        def _pack_stats(name: str, samples: np.ndarray) -> None:
            med = float(np.median(samples))
            sd = float(np.std(samples))
            low = float(np.percentile(samples, 16))
            high = float(np.percentile(samples, 84))
            result[name] = med
            result[f"{name}_err"] = sd
            result[f"{name}_16"] = low
            result[f"{name}_84"] = high

        # ── Fitted Parameters ─────────────────────────────────────────────────────
        _pack_stats("rp_r_star", rp_r_star_samples)
        _pack_stats("b", b_samples)
        _pack_stats("period", period_samples)
        _pack_stats("t0", t0_samples)
        _pack_stats("rho_star", rho_star_samples)

        # ── Derived Parameters ────────────────────────────────────────────────────
        # 1. Semi-major axis (a/R★)
        a_r_star_samples = (52.91572 * rho_star_samples * period_samples**2) ** (1.0 / 3.0)
        _pack_stats("a_r_star", a_r_star_samples)

        # 2. Transit duration T14 (hours)
        term_num = (1.0 + rp_r_star_samples) ** 2 - b_samples ** 2
        term_den = a_r_star_samples ** 2 - b_samples ** 2
        inside_arcsin = np.sqrt(np.maximum(term_num / np.maximum(term_den, 1e-5), 0.0))
        inside_arcsin = np.clip(inside_arcsin, 0.0, 1.0)
        t14_samples_hr = (period_samples / np.pi) * np.arcsin(inside_arcsin) * 24.0
        _pack_stats("t14_hr", t14_samples_hr)

        # Draw normal samples for stellar parameters to perform Monte Carlo propagation
        r_star_samples = np.random.normal(r_star, r_star_err, len(rp_r_star_samples))
        r_star_samples = np.maximum(r_star_samples, 0.01)  # prevent non-physical negative radius
        teff_samples = np.random.normal(teff, teff_err, len(rp_r_star_samples))
        teff_samples = np.maximum(teff_samples, 100.0)

        # 3. Planet Radius (Rp in Earth Radii)
        rp_earth_samples = rp_r_star_samples * r_star_samples * 109.203
        _pack_stats("rp_earth", rp_earth_samples)

        # 4. Semi-major axis in Astronomical Units (a in AU)
        a_au_samples = (a_r_star_samples * r_star_samples) / 215.032
        _pack_stats("a_au", a_au_samples)

        # 5. Equilibrium Temperature (Teq in K)
        teq_samples = teff_samples * (0.25 / a_r_star_samples**2) ** 0.25
        _pack_stats("t_eq", teq_samples)

        planets_list.append(result)

    return planets_list

