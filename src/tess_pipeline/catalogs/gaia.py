"""
catalogs/gaia.py — Query Gaia DR3 stellar parameters via astroquery.

Returns only values present in the Gaia DR3 catalog; never estimates
or fabricates missing stellar properties.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from tess_pipeline.exceptions import GaiaQueryError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)

_GAIA_RADIUS_DEG = 3.0 / 3600  # 3 arcsec search radius


def query_gaia(
    *,
    ra: float | None = None,
    dec: float | None = None,
    tic_id: int | None = None,
) -> dict[str, Any]:
    """
    Query Gaia DR3 for stellar parameters.

    At least one of *ra*/*dec* or *tic_id* must be provided.
    If only *tic_id* is given, coordinates are resolved from the TIC first.

    Returns
    -------
    dict with keys (None when not available in Gaia DR3):
        r_star      : stellar radius (R_Sun)
        r_star_err  : radius uncertainty (R_Sun)
        teff        : effective temperature (K)
        teff_err    : Teff uncertainty (K)
        lum         : luminosity (L_Sun)
        feh         : metallicity [Fe/H]
        logg        : surface gravity (dex)
        ra          : right ascension (deg)
        dec         : declination (deg)
        parallax    : parallax (mas)
        parallax_err: parallax uncertainty (mas)
        source_id   : Gaia DR3 source_id
    """
    if ra is None or dec is None:
        if tic_id is not None:
            ra, dec = _coords_from_tic(tic_id)
        if ra is None or dec is None:
            log.warning("No coordinates available for Gaia query; returning empty stellar dict")
            return _empty()

    try:
        return _do_query(ra, dec)
    except GaiaQueryError:
        raise
    except Exception as exc:
        log.warning("Gaia query failed: %s", exc)
        return _empty()


def _do_query(ra: float, dec: float) -> dict[str, Any]:
    """Execute the Gaia DR3 cone search."""
    try:
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError as exc:
        raise GaiaQueryError("astroquery is required for Gaia queries") from exc

    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    try:
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        job = Gaia.cone_search(coord, radius=u.Quantity(_GAIA_RADIUS_DEG, u.deg))
        table = job.get_results()
    except Exception as exc:
        raise GaiaQueryError(f"Gaia cone search failed: {exc}") from exc

    if table is None or len(table) == 0:
        log.info("No Gaia DR3 match within %.1f arcsec of (%.4f, %.4f)", 3.0, ra, dec)
        return _empty()

    # Use the closest match
    row = table[0]

    def _val(key: str) -> float | None:
        v = row.get(key)
        if v is None:
            return None
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def _half_range(upper: float | None, lower: float | None) -> float | None:
        """Convert Gaia asymmetric percentile bounds to a symmetric 1-sigma estimate."""
        if upper is None or lower is None:
            return None
        diff = upper - lower
        if diff < 0 or not math.isfinite(diff):
            return None
        return diff / 2.0

    result = {
        "r_star": _val("radius_val"),
        "r_star_err": _half_range(_val("radius_percentile_upper"), _val("radius_percentile_lower")),
        "teff": _val("teff_gspphot"),
        "teff_err": _half_range(_val("teff_gspphot_upper"), _val("teff_gspphot_lower")),
        "lum": _val("lum_gspphot"),
        "feh": _val("mh_gspphot"),
        "feh_err": _half_range(_val("mh_gspphot_upper"), _val("mh_gspphot_lower")),
        "logg": _val("logg_gspphot"),
        "logg_err": _half_range(_val("logg_gspphot_upper"), _val("logg_gspphot_lower")),
        "ra": _val("ra"),
        "dec": _val("dec"),
        "parallax": _val("parallax"),
        "parallax_err": _val("parallax_error"),
        "source_id": str(row.get("source_id", "")),
    }

    # Warn about missing critical fields
    for key in ("r_star", "teff", "parallax"):
        if result[key] is None:
            log.warning("Gaia DR3: %s not available for this source", key)

    return result


def _coords_from_tic(tic_id: int) -> tuple[float | None, float | None]:
    """Resolve TIC coordinates via lightkurve."""
    try:
        import lightkurve as lk

        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            return None, None
        t = search.table
        ra = float(t["s_ra"][0]) if "s_ra" in t.colnames else None
        dec = float(t["s_dec"][0]) if "s_dec" in t.colnames else None
        return ra, dec
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve coordinates for TIC %d: %s", tic_id, exc)
        return None, None


def _empty() -> dict[str, Any]:
    """Return an empty Gaia result dict."""
    return {
        "r_star": None,
        "r_star_err": None,
        "teff": None,
        "teff_err": None,
        "lum": None,
        "feh": None,
        "logg": None,
        "ra": None,
        "dec": None,
        "parallax": None,
        "parallax_err": None,
        "source_id": None,
    }
