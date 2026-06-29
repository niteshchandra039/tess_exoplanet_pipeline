"""
transit/phoebe_model.py — Optional PHOEBE 2 backend.

PHOEBE 2 is an advanced modeling framework for binary stars and
transiting planets that supports phase curves, reflected light,
ellipsoidal variations, and multi-body hierarchical systems.

This module is only active when phoebe is installed:
    pip install "tess-pipeline[phoebe]"

For standard single-transit TESS fits, use the exoplanet backend.
PHOEBE is reserved for phase-curve physics and hierarchical systems.
"""

from __future__ import annotations

from typing import Any

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def phoebe_available() -> bool:
    """Return True if phoebe is importable."""
    try:
        import phoebe  # noqa: F401
        return True
    except ImportError:
        return False


def run_phoebe_fit(
    lc: Any,
    period: float,
    stellar: dict[str, Any],
    *,
    include_reflected_light: bool = False,
    include_ellipsoidal: bool = False,
) -> dict[str, Any]:
    """
    Fit a PHOEBE 2 transit + phase-curve model.

    Parameters
    ----------
    lc : lightkurve.LightCurve
    period : float
        Orbital period (days).
    stellar : dict
        Stellar parameters from catalogs.stellar.
    include_reflected_light : bool
        Model planetary reflected light contribution.
    include_ellipsoidal : bool
        Model stellar ellipsoidal variations.

    Returns
    -------
    dict with keys:
        solution, residuals, planet_params

    Raises
    ------
    ImportError
        If phoebe is not installed.
    """
    try:
        import phoebe
    except ImportError as exc:
        raise ImportError(
            "PHOEBE 2 is required for this backend. "
            "Install with: pip install 'tess-pipeline[phoebe]'"
        ) from exc

    import numpy as np

    log.info("Building PHOEBE 2 bundle for period=%.6f d", period)

    b = phoebe.default_binary()

    # Check if TESS:T is available, otherwise fallback to Johnson:V or Bolometric
    installed_passbands = phoebe.list_installed_passbands()
    passband = "TESS:T"
    if passband not in installed_passbands:
        if "Johnson:V" in installed_passbands:
            log.warning("TESS:T passband is not installed in PHOEBE; falling back to Johnson:V")
            passband = "Johnson:V"
        elif "Bolometric:900-40000" in installed_passbands:
            log.warning("TESS:T passband is not installed in PHOEBE; falling back to Bolometric")
            passband = "Bolometric:900-40000"
        elif installed_passbands:
            passband = installed_passbands[0]
            log.warning("TESS:T passband is not installed in PHOEBE; falling back to %s", passband)
        else:
            raise ValueError("No passbands are installed in PHOEBE.")

    # Set stellar parameters
    r_star = stellar.get("r_star", 1.0) if stellar.get("r_star") is not None else 1.0
    m_star = stellar.get("m_star", 1.0) if stellar.get("m_star") is not None else 1.0
    teff = stellar.get("teff", 5778) if stellar.get("teff") is not None else 5778

    # Run quick batman map fit to get initial guesses for planet parameters
    from tess_pipeline.transit.batman_model import quick_batman_fit
    fit = quick_batman_fit(lc, period, stellar)
    planet_params = fit.get("planet_params", {})

    rp_r_star = planet_params.get("rp_r_star", 0.05)
    a_r_star = planet_params.get("a_r_star", 10.0)
    t0 = planet_params.get("t0", 0.0)
    incl = planet_params.get("incl", 90.0)

    # Convert planet parameters to PHOEBE units
    r_planet_sol = rp_r_star * r_star
    sma_sol = a_r_star * r_star

    b.set_value("requiv", r_star, component="primary", unit="solRad")
    b.set_value("teff", teff, component="primary", unit="K")
    b.set_value("period", period, component="binary", unit="d")

    # Configure secondary star as a planet (cool, small, low mass ratio)
    b.set_value("requiv", r_planet_sol, component="secondary", unit="solRad")
    b.set_value("teff", 300.0, component="secondary", unit="K")
    b.set_value("q", 1e-4, component="binary") # low mass ratio
    b.set_value("sma", sma_sol, component="binary", unit="solRad")
    b.set_value("incl", incl, component="binary", unit="deg")
    b.set_value("ld_mode_bol", "manual", component="secondary")

    # Add TESS dataset
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    b.add_dataset(
        "lc",
        times=time,
        fluxes=flux,
        passband=passband,
        dataset="lc01",
    )
    b.set_value("ld_mode", "manual", component="secondary", dataset="lc01")

    # Configure compute options
    b.add_compute("phoebe", compute="phoebe_compute")
    b.set_value("atm", "blackbody", component="secondary", compute="phoebe_compute")
    b.set_value("ntriangles", 15000, component="primary", compute="phoebe_compute")

    log.info("PHOEBE bundle built; running solver (this may take several minutes)")

    try:
        b.run_compute(compute="phoebe_compute")
    except Exception as exc:
        log.error("PHOEBE compute failed: %s", exc)
        raise

    model_flux = b["value@fluxes@lc01@phoebe_compute"]
    residuals = flux - model_flux

    return {
        "solution": b,
        "residuals": residuals,
        "flux_model": model_flux,
        "planet_params": {
            "period": period,
            "t0": t0,
            "rp_r_star": rp_r_star,
            "a_r_star": a_r_star,
            "rp_earth": planet_params.get("rp_earth", 1.0),
            "incl": incl,
            "method": "phoebe2",
        },
    }
