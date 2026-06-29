"""
catalogs/stellar.py — Stellar characterization via isochrone fitting.

Uses isoclassify (Huber et al. 2017) by default to derive homogeneous
Teff, logg, R★, M★, ρ★, L★, age with full uncertainties from
Gaia DR3 photometry, parallax, and (optionally) spectroscopic [Fe/H].

Follows the approach of Berger et al. 2023 (ApJS 264, 30).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)

# Solar constants used for unit conversions
_R_SUN = 6.957e10    # cm
_M_SUN = 1.989e33    # g
_L_SUN = 3.828e33    # erg/s


def characterize_star(
    gaia_params: dict[str, Any],
    *,
    method: str = "isoclassify",
) -> dict[str, Any]:
    """
    Derive stellar parameters from Gaia DR3 observables.

    Parameters
    ----------
    gaia_params : dict
        Output of ``catalogs.gaia.query_gaia()``.
    method : str
        "isoclassify" (default) or "gaia_only".

    Returns
    -------
    dict with keys (None when unavailable):
        r_star, r_star_err          : R★ (R_Sun)
        m_star, m_star_err          : M★ (M_Sun)
        teff, teff_err              : Teff (K)
        logg, logg_err              : log g (dex)
        feh, feh_err                : [Fe/H]
        lum, lum_err                : L★ (L_Sun)
        rho_star, rho_star_err      : ρ★ (g/cm³)
        age, age_err                : age (Gyr)
        method                      : characterization method used
    """
    if method == "isoclassify":
        result = _run_isoclassify(gaia_params)
        if result is not None:
            return result
        log.warning("isoclassify failed; falling back to Gaia-only parameters")

    return _gaia_only(gaia_params)


def _run_isoclassify(gaia_params: dict[str, Any]) -> dict[str, Any] | None:
    """Run isoclassify direct method using Gaia inputs."""
    try:
        import isoclassify
        from isoclassify.direct import classify
    except ImportError:
        log.warning(
            "isoclassify is not installed. "
            "Install with: pip install 'tess-pipeline[stellar]'. "
            "Falling back to Gaia-only parameters."
        )
        return None

    # Build isoclassify input dict
    r_star = gaia_params.get("r_star")
    teff = gaia_params.get("teff")
    parallax = gaia_params.get("parallax")
    parallax_err = gaia_params.get("parallax_err")
    feh = gaia_params.get("feh")

    if r_star is None or teff is None or parallax is None:
        log.warning(
            "Insufficient Gaia parameters for isoclassify (need r_star, teff, parallax)"
        )
        return None

    try:
        # isoclassify direct mode inputs
        inputs = {
            "teff": teff,
            "teff_err": gaia_params.get("teff_err", 100.0) or 100.0,
            "parallax": parallax,
            "parallax_err": parallax_err or 0.1,
            "feh": feh if feh is not None else 0.0,
            "feh_err": 0.1,
        }
        iso_result = classify(**inputs)

        result: dict[str, Any] = {
            "r_star": _safe(iso_result.get("rad")),
            "r_star_err": _safe(iso_result.get("rad_err")),
            "m_star": _safe(iso_result.get("mass")),
            "m_star_err": _safe(iso_result.get("mass_err")),
            "teff": _safe(iso_result.get("teff")) or teff,
            "teff_err": _safe(iso_result.get("teff_err")),
            "logg": _safe(iso_result.get("logg")),
            "logg_err": _safe(iso_result.get("logg_err")),
            "feh": _safe(iso_result.get("feh")) or feh,
            "feh_err": _safe(iso_result.get("feh_err")),
            "lum": _safe(iso_result.get("lum")),
            "lum_err": _safe(iso_result.get("lum_err")),
            "age": _safe(iso_result.get("age")),
            "age_err": _safe(iso_result.get("age_err")),
            "method": "isoclassify",
        }
        result["rho_star"], result["rho_star_err"] = _compute_rho(
            result.get("m_star"), result.get("r_star"),
            result.get("m_star_err"), result.get("r_star_err"),
        )
        return result

    except Exception as exc:
        log.warning("isoclassify.classify failed: %s", exc)
        return None


def _gaia_only(gaia_params: dict[str, Any]) -> dict[str, Any]:
    """Return stellar parameters directly from Gaia DR3 (no isochrone fitting)."""
    r_star = gaia_params.get("r_star")
    teff = gaia_params.get("teff")

    if r_star is None:
        log.warning("Gaia R★ unavailable; stellar characterization incomplete")
    if teff is None:
        log.warning("Gaia Teff unavailable; stellar characterization incomplete")

    result: dict[str, Any] = {
        "r_star": r_star,
        "r_star_err": gaia_params.get("r_star_err"),
        "m_star": None,       # mass not available from Gaia alone
        "m_star_err": None,
        "teff": teff,
        "teff_err": gaia_params.get("teff_err"),
        "logg": gaia_params.get("logg"),
        "logg_err": None,
        "feh": gaia_params.get("feh"),
        "feh_err": None,
        "lum": gaia_params.get("lum"),
        "lum_err": None,
        "age": None,
        "age_err": None,
        "method": "gaia_only",
    }
    result["rho_star"], result["rho_star_err"] = _compute_rho(
        None, r_star, None, gaia_params.get("r_star_err")
    )
    return result


def _compute_rho(
    m_star: float | None,
    r_star: float | None,
    m_err: float | None,
    r_err: float | None,
) -> tuple[float | None, float | None]:
    """
    Compute stellar density ρ★ in g/cm³ from M★ and R★ in solar units.

    Returns (rho, rho_err); both None if inputs are unavailable.
    """
    if m_star is None or r_star is None or r_star <= 0:
        return None, None

    m_cgs = m_star * _M_SUN
    r_cgs = r_star * _R_SUN
    rho = m_cgs / (4.0 / 3.0 * math.pi * r_cgs**3)

    rho_err: float | None = None
    if m_err is not None and r_err is not None:
        # Gaussian error propagation: σρ/ρ = sqrt((σM/M)² + (3σR/R)²)
        rho_err = rho * math.sqrt((m_err / m_star) ** 2 + (3 * r_err / r_star) ** 2)

    return rho, rho_err


def _safe(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
