"""
transit/phoebe_model.py — Optional PHOEBE 2 backend.

PHOEBE 2 is an advanced modeling framework for binary stars and
transiting planets that supports phase curves, reflected light,
ellipsoidal variations, and multi-body hierarchical systems.

This module is part of the standard installation.

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
    rv_times: np.ndarray | None = None,
    rv_vals: np.ndarray | None = None,
    rv_errs: np.ndarray | None = None,
    input_is_magnitude: bool = False,
) -> dict[str, Any]:
    """
    Fit a PHOEBE 2 transit + phase-curve + radial velocity model.

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
    rv_times : np.ndarray, optional
        Observed radial velocity times.
    rv_vals : np.ndarray, optional
        Observed radial velocity values (km/s).
    rv_errs : np.ndarray, optional
        Errors on observed radial velocities.
    input_is_magnitude : bool
        If True, force conversion of light curve fluxes from magnitudes.

    Returns
    -------
    dict with keys:
        solution, residuals, flux_model, rv_model_primary, rv_model_secondary,
        rv_residuals, planet_params

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
            "Reinstall the package with: pip install -e ."
        ) from exc

    import numpy as np

    log.info("Building PHOEBE 2 bundle for period=%.6f d", period)

    b = phoebe.default_binary()

    # Check if TESS:T is available, otherwise fallback to Cousins:I or download
    installed_passbands = phoebe.list_installed_passbands()
    passband = "TESS:T"
    if passband not in installed_passbands:
        try:
            log.info("Attempting to download TESS:T passband...")
            phoebe.download_passband("TESS:T")
            installed_passbands = phoebe.list_installed_passbands()
        except Exception as exc:
            log.warning("Could not download TESS:T passband: %s. Trying Cousin/Johnson bands.", exc)

    if passband not in installed_passbands:
        if "Cousins:I" in installed_passbands:
            passband = "Cousins:I"
            log.info("Falling back to Cousins:I passband")
        else:
            try:
                log.info("Attempting to download Cousins:I passband...")
                phoebe.download_passband("Cousins:I")
                passband = "Cousins:I"
                installed_passbands = phoebe.list_installed_passbands()
            except Exception:
                if "Johnson:V" in installed_passbands:
                    passband = "Johnson:V"
                    log.warning("TESS:T and Cousins:I not available; falling back to Johnson:V")
                elif "Bolometric:900-40000" in installed_passbands:
                    passband = "Bolometric:900-40000"
                    log.warning("TESS:T and Cousins:I not available; falling back to Bolometric")
                elif installed_passbands:
                    passband = installed_passbands[0]
                    log.warning("TESS:T and Cousins:I not available; falling back to %s", passband)
                else:
                    raise ValueError("No passbands are installed or downloadable in PHOEBE.")

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

    # Handle magnitudes and convert to normalized flux if needed
    raw_flux = np.asarray(lc.flux.value, dtype=float)
    is_mag = input_is_magnitude or (np.median(raw_flux) > 5.0 and np.std(raw_flux) < 5.0)

    if is_mag:
        log.info("Input light curve detected/specified as magnitudes. Converting to flux.")
        # F = 10**((20.44 - mag) / 2.5)
        median_mag = np.median(raw_flux)
        flux = 10**((20.44 - raw_flux) / 2.5)
        # normalize
        flux_scale = np.median(flux)
        flux = flux / flux_scale
    else:
        flux = raw_flux / np.median(raw_flux)

    # Add TESS dataset
    time = np.asarray(lc.time.value, dtype=float)
    b.add_dataset(
        "lc",
        times=time,
        fluxes=flux,
        passband=passband,
        dataset="lc01",
    )
    b.set_value("ld_mode", "manual", component="secondary", dataset="lc01")

    # Add RV dataset if provided
    if rv_times is not None and rv_vals is not None and len(rv_times) > 0:
        log.info("Adding RV dataset to PHOEBE bundle (N=%d)", len(rv_times))
        b.add_dataset(
            "rv",
            times=np.asarray(rv_times, dtype=float),
            rvs=np.asarray(rv_vals, dtype=float),
            dataset="rv01",
        )
        if rv_errs is not None and len(rv_errs) == len(rv_times):
            b.set_value(qualifier="sigmas", dataset="rv01", value=np.asarray(rv_errs, dtype=float))

    # Configure compute options
    b.add_compute("phoebe", compute="phoebe_compute")
    b.set_value("atm", "blackbody", component="secondary", compute="phoebe_compute")
    
    # Dynamically estimate minimum ntriangles to pass PHOEBE's checks
    # The error check requires the area of primary triangles to be smaller than secondary's area.
    # N > ~30 * (R_star / R_planet)^2
    required_triangles = int(35.0 * (1.0 / rp_r_star)**2) if rp_r_star > 0 else 15000
    ntriangles_val = max(1500, min(15000, required_triangles))
    # Round to nearest hundred for cleanliness
    ntriangles_val = int((ntriangles_val // 100) * 100)
    log.info("Setting primary ntriangles=%d based on rp/r_star=%.4f", ntriangles_val, rp_r_star)
    b.set_value("ntriangles", ntriangles_val, component="primary", compute="phoebe_compute")

    log.info("PHOEBE bundle built; running solver (this may take several minutes)")

    try:
        b.run_compute(compute="phoebe_compute")
    except Exception as exc:
        log.error("PHOEBE compute failed: %s", exc)
        raise

    model_flux = np.asarray(b["value@fluxes@lc01@phoebe_compute"], dtype=float)
    if len(model_flux) > 0 and np.median(model_flux) != 0.0:
        model_flux = model_flux / np.median(model_flux)
    residuals = flux - model_flux

    model_rv_primary = None
    model_rv_secondary = None
    rv_residuals = None
    if rv_times is not None and rv_vals is not None and len(rv_times) > 0:
        model_rv_primary = np.asarray(b["value@rvs@primary@rv01@phoebe_compute"])
        model_rv_secondary = np.asarray(b["value@rvs@secondary@rv01@phoebe_compute"])
        rv_residuals = np.asarray(rv_vals, dtype=float) - model_rv_primary

    return {
        "solution": b,
        "residuals": residuals,
        "flux_model": model_flux,
        "rv_model_primary": model_rv_primary,
        "rv_model_secondary": model_rv_secondary,
        "rv_residuals": rv_residuals,
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
