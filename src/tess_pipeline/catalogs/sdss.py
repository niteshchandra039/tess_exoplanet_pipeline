"""
catalogs/sdss.py — Query SDSS spectra for radial velocity measurements.

Coverage note: SDSS spectroscopic coverage is incomplete. Many TESS targets
will have no SDSS match. The pipeline proceeds without RV in those cases.
Never fabricates values; returns None when unavailable.
"""

from __future__ import annotations

import math
from typing import Any

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)

_SEARCH_RADIUS_DEG = 3.0 / 3600   # 3 arcsec


def query_sdss_rv(
    *,
    ra: float | None,
    dec: float | None,
) -> dict[str, Any] | None:
    """
    Query SDSS DR18 for a radial velocity measurement near (*ra*, *dec*).

    Returns
    -------
    dict with keys ``rv`` (km/s) and ``rv_err`` (km/s), or ``None`` if
    no spectroscopic match is found or RV is unavailable.
    """
    if ra is None or dec is None:
        log.info("No coordinates for SDSS query; skipping")
        return None

    try:
        return _do_query(ra, dec)
    except Exception as exc:  # noqa: BLE001
        log.warning("SDSS RV query failed: %s", exc)
        return None


def _do_query(ra: float, dec: float) -> dict[str, Any] | None:
    try:
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError as exc:
        log.warning("astroquery is required for SDSS queries: %s", exc)
        return None

    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    try:
        table = SDSS.query_region(
            coord,
            radius=_SEARCH_RADIUS_DEG * u.deg,
            spectro=True,
            photoobj_fields=None,
            specobj_fields=["ra", "dec", "velDisp", "z", "zErr"],
        )
    except Exception as exc:
        log.info("SDSS spectroscopic search returned no results: %s", exc)
        return None

    if table is None or len(table) == 0:
        log.info("No SDSS spectroscopic match within 3 arcsec")
        return None

    row = table[0]
    # Convert redshift z → radial velocity (non-relativistic approximation)
    z = _safe_float(row.get("z"))
    z_err = _safe_float(row.get("zErr"))

    if z is None:
        log.info("SDSS match found but redshift unavailable")
        return None

    c_kms = 299792.458   # speed of light in km/s
    rv = z * c_kms
    rv_err = z_err * c_kms if z_err is not None else None

    log.info("SDSS RV: %.2f ± %.2f km/s", rv, rv_err or float("nan"))
    return {"rv": rv, "rv_err": rv_err}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
