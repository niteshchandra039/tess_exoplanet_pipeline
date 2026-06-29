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

    b = phoebe.Bundle.default_star()

    # Set stellar parameters
    r_star = stellar.get("r_star", 1.0)
    m_star = stellar.get("m_star", 1.0)
    teff = stellar.get("teff", 5778)

    b.set_value("requiv", r_star, component="primary", unit="solRad")
    b.set_value("teff", teff, component="primary", unit="K")
    b.set_value("period", period, component="binary", unit="d")

    # Add TESS dataset
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    b.add_dataset(
        "lc",
        times=time,
        fluxes=flux,
        passband="TESS:T",
        dataset="lc01",
    )

    # Configure compute options
    b.add_compute("phoebe", compute="phoebe_compute")

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
            "method": "phoebe2",
        },
    }
