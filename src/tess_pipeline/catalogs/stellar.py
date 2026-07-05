"""
catalogs/stellar.py — Stellar characterization from multi-catalog merging.

Derives Teff, logg, R★, M★, ρ★ by merging three authoritative catalogs
in strict priority order:
    1. VizieR – TIC v8.2  (Paegert et al. 2021, IV/39/tic82)
    2. Gaia DR3            (Gaia Collaboration 2022)
    3. SIMBAD              (latest bibcode by year)

All parameters carry source, reference (bibcode/DOI), and unit metadata
for full provenance tracking in downstream Bayesian transit modeling.
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
    method: str = "gaia_only",
) -> dict[str, Any]:
    """
    Derive stellar parameters from Gaia DR3, VizieR (TIC v8.2), and SIMBAD.

    Priority order:
        1. VizieR (TIC v8.2)
        2. Gaia DR3
        3. SIMBAD (latest value)
    """
    # 1. Fetch parameters from all three sources
    tic_id = gaia_params.get("tic_id")
    ra = gaia_params.get("ra")
    dec = gaia_params.get("dec")

    # Query VizieR
    vizier_params = _query_vizier_tic(tic_id) if tic_id is not None else None

    # Query SIMBAD
    simbad_params = _query_simbad(tic_id=tic_id, ra=ra, dec=dec)

    # 2. Merge parameter-by-parameter based on priority
    merged_params = {}
    adopted_sources = {}

    # Add non-parameter metadata
    merged_params["tic_id"] = tic_id
    merged_params["ra"] = ra
    merged_params["dec"] = dec

    # Generic merge for all parameters except feh: VizieR → Gaia → SIMBAD
    parameters_to_merge = ["r_star", "m_star", "teff", "logg", "parallax", "lum"]

    for k in parameters_to_merge:
        # Priority 1: VizieR (TIC v8.2)
        if vizier_params is not None and vizier_params.get(k) is not None:
            merged_params[k] = vizier_params[k]
            merged_params[f"{k}_err"] = vizier_params.get(f"{k}_err")
            merged_params[f"{k}_ref"] = vizier_params.get(f"{k}_ref") or "TIC v8.2"
            merged_params[f"{k}_unit"] = vizier_params.get(f"{k}_unit")
            adopted_sources[k] = "VizieR (TIC8.2)"

        # Priority 2: Gaia DR3
        elif gaia_params.get(k) is not None:
            merged_params[k] = gaia_params[k]
            merged_params[f"{k}_err"] = gaia_params.get(f"{k}_err")
            merged_params[f"{k}_ref"] = "Gaia DR3"
            # Set default units
            if k == "teff":
                merged_params["teff_unit"] = "K"
            elif k == "logg":
                merged_params["logg_unit"] = "dex"
            elif k == "parallax":
                merged_params["parallax_unit"] = "mas"
            elif k == "r_star":
                merged_params["r_star_unit"] = "R_sun"
            elif k == "m_star":
                merged_params["m_star_unit"] = "M_sun"
            elif k == "lum":
                merged_params["lum_unit"] = "L_sun"
            adopted_sources[k] = "Gaia"

        # Priority 3: SIMBAD (latest value)
        elif simbad_params is not None and simbad_params.get(k) is not None:
            merged_params[k] = simbad_params[k]
            merged_params[f"{k}_err"] = simbad_params.get(f"{k}_err")
            merged_params[f"{k}_ref"] = simbad_params.get(f"{k}_ref")
            merged_params[f"{k}_unit"] = simbad_params.get(f"{k}_unit")
            adopted_sources[k] = "SIMBAD"

    # [Fe/H] — special priority: SIMBAD (spectroscopic) → VizieR → Gaia
    # SIMBAD aggregates peer-reviewed spectroscopic surveys and selects the
    # most recent measurement by bibcode year, which is more reliable than
    # Gaia's photometric [M/H] (mh_gspphot).
    if simbad_params is not None and simbad_params.get("feh") is not None:
        merged_params["feh"] = simbad_params["feh"]
        merged_params["feh_err"] = simbad_params.get("feh_err")
        merged_params["feh_ref"] = simbad_params.get("feh_ref")
        merged_params["feh_unit"] = simbad_params.get("feh_unit") or "dex"
        adopted_sources["feh"] = "SIMBAD"
    elif vizier_params is not None and vizier_params.get("feh") is not None:
        merged_params["feh"] = vizier_params["feh"]
        merged_params["feh_err"] = vizier_params.get("feh_err")
        merged_params["feh_ref"] = vizier_params.get("feh_ref") or "TIC v8.2"
        merged_params["feh_unit"] = vizier_params.get("feh_unit") or "dex"
        adopted_sources["feh"] = "VizieR (TIC8.2)"
    elif gaia_params.get("feh") is not None:
        merged_params["feh"] = gaia_params["feh"]
        merged_params["feh_err"] = gaia_params.get("feh_err")
        merged_params["feh_ref"] = "Gaia DR3"
        merged_params["feh_unit"] = "dex"
        adopted_sources["feh"] = "Gaia"

    # If m_star is missing, estimate it using Torres et al. 2010
    if merged_params.get("m_star") is None:
        teff_val = merged_params.get("teff")
        r_val = merged_params.get("r_star")
        if teff_val is not None and r_val is not None:
            merged_params["m_star"] = (teff_val / 5777.0)**1.5 * r_val**0.1
            merged_params["m_star_err"] = None
            merged_params["m_star_ref"] = "Torres et al. (2010)"
            merged_params["m_star_unit"] = "M_sun"
            adopted_sources["m_star"] = "Torres et al. 2010"

    # Adopt catalog parameters directly (VizieR > Gaia > SIMBAD priority already merged above)
    result = _gaia_only(merged_params)

    # Set provenance reference
    ref_list = []
    for k in ("r_star", "teff", "logg", "feh", "parallax"):
        ref = merged_params.get(f"{k}_ref")
        if ref and ref not in ref_list:
            ref_list.append(ref)
    result["reference"] = "+".join(ref_list) if ref_list else "Catalog"

    # Add source and reference metadata for each adopted parameter
    for k, src in adopted_sources.items():
        result[f"{k}_source"] = src
        ref = merged_params.get(f"{k}_ref")
        if ref:
            result[f"{k}_ref"] = ref
        unit = merged_params.get(f"{k}_unit")
        if unit:
            result[f"{k}_unit"] = unit

    if result.get("r_star") is None or result.get("m_star") is None:
        raise ValueError("Stellar radius and mass could not be resolved from VizieR, Gaia, or SIMBAD. Physical transit modeling requires these parameters.")

    return result





def _gaia_only(params: dict[str, Any]) -> dict[str, Any]:
    """Return stellar parameters directly from merged parameters."""
    r_star = params.get("r_star")
    teff = params.get("teff")

    if r_star is None:
        raise ValueError("Stellar radius (R★) is unavailable; cannot proceed with transit modeling.")
    if teff is None:
        raise ValueError("Stellar effective temperature (Teff) is unavailable; cannot proceed with transit modeling.")

    m_star = params.get("m_star")
    m_star_err = params.get("m_star_err")
    r_star_err = params.get("r_star_err")

    result: dict[str, Any] = {
        "r_star": r_star,
        "r_star_err": r_star_err,
        "m_star": m_star,
        "m_star_err": m_star_err,
        "teff": teff,
        "teff_err": params.get("teff_err"),
        "logg": params.get("logg"),
        "logg_err": params.get("logg_err"),
        "feh": params.get("feh"),
        "feh_err": params.get("feh_err"),
        "lum": params.get("lum"),
        "lum_err": params.get("lum_err"),
        "parallax": params.get("parallax"),
        "parallax_err": params.get("parallax_err"),
        "age": None,
        "age_err": None,
        "method": "gaia_only",
        "r_star_unit": params.get("r_star_unit") or "R_sun",
        "m_star_unit": params.get("m_star_unit") or "M_sun",
        "rho_star_unit": params.get("rho_star_unit") or "g/cm^3",
    }
    result["rho_star"], result["rho_star_err"] = _compute_rho(
        m_star, r_star, m_star_err, r_star_err
    )
    return result


# Ordered list of VizieR mirror hostnames to try in sequence.
# astroquery.vizier.conf.server accepts a bare hostname (no protocol/path).
# If the primary CDS server is unreachable or slow, the query is
# automatically retried on the next available mirror.
_VIZIER_MIRRORS = [
    "vizier.cds.unistra.fr",    # primary (Strasbourg, France)
    "vizier.iucaa.in",          # India (IUCAA, Pune)
    "vizier.nao.ac.jp",         # Japan (NAOJ, Tokyo)
    "vizier.hia.nrc.ca",        # Canada (HIA)
]


def _query_vizier_tic(tic_id: int) -> dict[str, Any] | None:
    """Query VizieR TIC v8.2 (IV/39/tic82) for stellar parameters.

    Tries each mirror in ``_VIZIER_MIRRORS`` in order and returns the
    first successful result.  Falls back to ``None`` only if every
    mirror fails.
    """
    try:
        from astroquery.vizier import Vizier
        from astroquery.vizier import conf as vizier_conf

        last_exc: Exception | None = None
        result_list = None

        for server in _VIZIER_MIRRORS:
            try:
                log.debug("Querying VizieR TIC v8.2 via %s", server)
                with vizier_conf.set_temp("server", server):
                    v = Vizier(catalog="IV/39/tic82", columns=["*"])
                    result_list = v.query_constraints(TIC=str(tic_id))
                if result_list and len(result_list) > 0:
                    log.info("VizieR TIC v8.2 query succeeded via %s", server)
                    break
                # Empty result from this mirror — try the next
                result_list = None
            except Exception as exc:
                log.warning("VizieR mirror %s failed: %s — trying next mirror", server, exc)
                last_exc = exc
                result_list = None

        if result_list is None or len(result_list) == 0:
            if last_exc is not None:
                log.warning("All VizieR mirrors exhausted. Last error: %s", last_exc)
            return None

        table = result_list[0]
        if len(table) == 0:
            return None

        row = table[0]

        # Helper to extract value
        def _val(col_name: str) -> float | None:
            actual_col = None
            for c in row.colnames:
                if c.lower() == col_name.lower():
                    actual_col = c
                    break
            if actual_col is None:
                return None
            val = row[actual_col]
            if val is None or str(val) in ("--", "nan", "NaN", ""):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        # Helper to extract string
        def _str(col_name: str) -> str | None:
            actual_col = None
            for c in row.colnames:
                if c.lower() == col_name.lower():
                    actual_col = c
                    break
            if actual_col is None:
                return None
            val = row[actual_col]
            if val is None or str(val) in ("--", "nan", "NaN", ""):
                return None
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return str(val)

        # Helper to extract unit
        def _unit(col_name: str) -> str | None:
            actual_col = None
            for c in row.colnames:
                if c.lower() == col_name.lower():
                    actual_col = c
                    break
            if actual_col is None:
                return None
            col_obj = table[actual_col]
            if hasattr(col_obj, "unit") and col_obj.unit:
                u_str = str(col_obj.unit).replace(" ", "").replace("**", "")
                if u_str == "unit-degK":
                    return "K"
                if u_str in ("cm/s2", "cm/s**2"):
                    return "dex"
                return u_str
            return None

        # Extract values
        teff = _val("teff")
        teff_err = _val("e_teff") or _val("eneg_teff") or _val("epos_teff")
        teff_ref = _str("teffsrc") or "TIC v8.2"
        teff_unit = _unit("teff") or "K"

        logg = _val("logg")
        logg_err = _val("e_logg")
        logg_ref = _str("loggsrc") or "TIC v8.2"
        logg_unit = _unit("logg") or "dex"

        feh = _val("MH") or _val("feh")
        feh_err = _val("e_MH") or _val("e_feh")
        feh_ref = _str("MHsrc") or _str("fehsrc") or "TIC v8.2"
        feh_unit = _unit("MH") or _unit("feh") or "dex"

        r_star = _val("rad")
        r_star_err = _val("e_rad")
        r_star_ref = _str("radsrc") or "TIC v8.2"
        r_star_unit = _unit("rad") or "R_sun"

        m_star = _val("mass")
        m_star_err = _val("e_mass")
        m_star_ref = _str("masssrc") or "TIC v8.2"
        m_star_unit = _unit("mass") or "M_sun"

        parallax = _val("plx")
        parallax_err = _val("e_plx")
        parallax_ref = _str("plxsrc") or "TIC v8.2"
        parallax_unit = _unit("plx") or "mas"

        viz_params = {
            "teff": teff,
            "teff_err": teff_err,
            "teff_ref": teff_ref,
            "teff_unit": teff_unit,
            "logg": logg,
            "logg_err": logg_err,
            "logg_ref": logg_ref,
            "logg_unit": logg_unit,
            "feh": feh,
            "feh_err": feh_err,
            "feh_ref": feh_ref,
            "feh_unit": feh_unit,
            "r_star": r_star,
            "r_star_err": r_star_err,
            "r_star_ref": r_star_ref,
            "r_star_unit": r_star_unit,
            "m_star": m_star,
            "m_star_err": m_star_err,
            "m_star_ref": m_star_ref,
            "m_star_unit": m_star_unit,
            "parallax": parallax,
            "parallax_err": parallax_err,
            "parallax_ref": parallax_ref,
            "parallax_unit": parallax_unit,
            "source": "VizieR (TIC v8.2)",
        }

        if any(v is not None for v in (teff, logg, feh, r_star, m_star, parallax)):
            log.info("VizieR (TIC8.2) query successful: Teff=%s, logg=%s, feh=%s, r_star=%s, m_star=%s, parallax=%s", teff, logg, feh, r_star, m_star, parallax)
            return viz_params

        return None
    except Exception as exc:
        log.warning("VizieR TIC query failed: %s", exc)
        return None


def _compute_rho(
    m_star: float | None,
    r_star: float | None,
    m_err: float | None,
    r_err: float | None,
) -> tuple[float | None, float | None]:
    """
    Compute stellar density ρ★ in g/cm³ from M★ and R★ in solar units.

    Formula:
        rho = rho_sun * (M★ / R★³)
    Source/Reference:
        IAU 2015 Resolution B3 (Prša et al. 2016, AJ 152, 41) defining nominal solar constants:
        - Nominal Solar Mass Parameter: G*M_sun = 1.3271244e20 m³/s²
        - Nominal Solar Radius: R_sun = 6.957e8 m
        - Newtonian Gravity Constant: G = 6.6743e-11 m³/(kg*s²)
        - Resulting Nominal Solar Density: rho_sun ≈ 1.4098 g/cm³ (precisely 1.4098418 g/cm³)
    """
    if m_star is None or r_star is None or r_star <= 0:
        return None, None

    rho_sun = 1.40984
    rho = rho_sun * m_star / (r_star**3)

    rho_err: float | None = None
    if m_err is not None and r_err is not None:
        # Gaussian error propagation: σρ/ρ = sqrt((σM/M)² + (3σR/R)²)
        rho_err = rho * math.sqrt((m_err / m_star) ** 2 + (3.0 * r_err / r_star) ** 2)

    return rho, rho_err


def _safe(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _query_simbad(
    tic_id: int | None = None,
    ra: float | None = None,
    dec: float | None = None,
) -> dict[str, Any] | None:
    """Query SIMBAD for stellar parameters as a fallback."""
    try:
        from astroquery.simbad import Simbad
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        # Enable the necessary fields on the Simbad instance
        Simbad.add_votable_fields("fe_h", "plx")

        # Initialize default values
        teff, logg, feh, parallax = None, None, None, None
        teff_bib, logg_bib, feh_bib, parallax_bib = None, None, None, None
        teff_unit, logg_unit, feh_unit, parallax_unit = None, None, None, None

        # Helper to extract the 4-digit year from a bibcode
        def _get_year(bibcode: str | None) -> int:
            if not bibcode:
                return 0
            try:
                year_str = bibcode[:4]
                if year_str.isdigit():
                    return int(year_str)
            except Exception:
                pass
            return 0

        # 1. Try SIMBAD TAP query for latest measurements
        tap_table = None
        if tic_id is not None:
            log.info("Querying SIMBAD TAP for TIC %d measurements", tic_id)
            try:
                query = f"""
                SELECT fe_h, teff, log_g, bibcode
                FROM mesfe_h
                WHERE oidref IN (
                    SELECT oidref FROM ident WHERE id = 'TIC {tic_id}'
                )
                """
                tap_table = Simbad.query_tap(query)
            except Exception as e:
                log.debug("SIMBAD TAP query by TIC ID failed: %s", e)

        if (tap_table is None or len(tap_table) == 0) and ra is not None and dec is not None:
            log.info("Querying SIMBAD TAP at coordinates RA=%s, DEC=%s for measurements", ra, dec)
            try:
                query = f"""
                SELECT m.fe_h, m.teff, m.log_g, m.bibcode
                FROM mesfe_h m
                JOIN basic b ON b.oid = m.oidref
                WHERE contains(point('ICRS', b.ra, b.dec), circle('ICRS', {ra}, {dec}, 0.000833)) = 1
                """
                tap_table = Simbad.query_tap(query)
            except Exception as e:
                log.debug("SIMBAD TAP query by coordinates failed: %s", e)

        if tap_table is not None and len(tap_table) > 0:
            def _get_val(row_data, col):
                if col not in tap_table.colnames:
                    return None
                v = row_data[col]
                if v is None or str(v) in ("--", "nan", "NaN", ""):
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            best_teff_year = 0
            best_logg_year = 0
            best_feh_year = 0

            for r in tap_table:
                bib = r["bibcode"]
                if isinstance(bib, bytes):
                    bib = bib.decode("utf-8")
                bib_str = str(bib) if bib else None
                year = _get_year(bib_str)

                # Process Teff
                t_val = _get_val(r, "teff")
                if t_val is not None:
                    if year > best_teff_year or teff is None:
                        teff = t_val
                        teff_bib = bib_str
                        best_teff_year = max(year, 1)
                        teff_unit = "K"
                        col_obj = tap_table["teff"]
                        if hasattr(col_obj, "unit") and col_obj.unit:
                            u_str = str(col_obj.unit).replace(" ", "").replace("**", "")
                            if u_str == "unit-degK":
                                teff_unit = "K"
                            else:
                                teff_unit = u_str

                # Process logg
                g_val = _get_val(r, "log_g")
                if g_val is not None:
                    if year > best_logg_year or logg is None:
                        logg = g_val
                        logg_bib = bib_str
                        best_logg_year = max(year, 1)
                        logg_unit = "dex"
                        col_obj = tap_table["log_g"]
                        if hasattr(col_obj, "unit") and col_obj.unit:
                            u_str = str(col_obj.unit).replace(" ", "").replace("**", "")
                            if u_str in ("cm/s2", "cm/s**2"):
                                logg_unit = "dex"
                            else:
                                logg_unit = u_str

                # Process feh
                f_val = _get_val(r, "fe_h")
                if f_val is not None:
                    if year > best_feh_year or feh is None:
                        feh = f_val
                        feh_bib = bib_str
                        best_feh_year = max(year, 1)
                        feh_unit = "dex"
                        col_obj = tap_table["fe_h"]
                        if hasattr(col_obj, "unit") and col_obj.unit:
                            feh_unit = str(col_obj.unit)

        # 2. Query basic object info for parallax and fallback values
        main_table = None
        if tic_id is not None:
            try:
                main_table = Simbad.query_object(f"TIC {tic_id}")
            except Exception:
                pass

        if (main_table is None or len(main_table) == 0) and ra is not None and dec is not None:
            try:
                coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
                main_table = Simbad.query_region(coord, radius=3 * u.arcsec)
            except Exception:
                pass

        if main_table is not None and len(main_table) > 0:
            main_row = main_table[0]

            def _main_val(col_name: str) -> float | None:
                if col_name not in main_table.colnames:
                    return None
                val = main_row[col_name]
                if val is None or str(val) in ("--", "nan", "NaN", ""):
                    return None
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            def _main_str(col_name: str) -> str | None:
                if col_name not in main_table.colnames:
                    return None
                val = main_row[col_name]
                if val is None or str(val) in ("--", "nan", "NaN", ""):
                    return None
                if isinstance(val, bytes):
                    return val.decode("utf-8")
                return str(val)

            def _main_unit(col_name: str) -> str | None:
                col_obj = main_table[col_name]
                if hasattr(col_obj, "unit") and col_obj.unit:
                    u_str = str(col_obj.unit).replace(" ", "").replace("**", "")
                    if u_str == "unit-degK":
                        return "K"
                    if u_str in ("cm/s2", "cm/s**2"):
                        return "dex"
                    return u_str
                return None

            # Parse parallax
            for col in main_table.colnames:
                col_lower = col.lower()
                if col_lower in ("plx", "parallax", "plx_value") or col_lower.endswith(".plx") or col_lower.endswith(".parallax"):
                    p_val = _main_val(col)
                    if p_val is not None:
                        parallax = p_val
                        parallax_unit = _main_unit(col) or "mas"
                elif col_lower == "coo_bibcode" or (col_lower.endswith(".bibcode") and ("plx" in col_lower or "parallax" in col_lower)):
                    p_bib = _main_str(col)
                    if p_bib:
                        parallax_bib = p_bib

            # Fallbacks for teff, logg, feh if TAP was empty or didn't find them
            for col in main_table.colnames:
                col_lower = col.lower()
                if "err" in col_lower or "qual" in col_lower:
                    continue

                if col_lower == "teff" or col_lower.endswith(".teff"):
                    if teff is None:
                        teff = _main_val(col)
                        teff_unit = _main_unit(col) or "K"
                elif col_lower in ("logg", "log_g") or col_lower.endswith(".log_g") or col_lower.endswith(".logg"):
                    if logg is None:
                        logg = _main_val(col)
                        logg_unit = _main_unit(col) or "dex"
                elif col_lower in ("fe_h", "feh") or col_lower.endswith(".fe_h") or col_lower.endswith(".feh"):
                    if feh is None:
                        feh = _main_val(col)
                        feh_unit = _main_unit(col) or "dex"

                # Bibcodes fallback
                if col_lower.endswith(".bibcode") or col_lower.endswith("_bibcode"):
                    val = _main_str(col)
                    if val:
                        if "fe_h" in col_lower or "feh" in col_lower:
                            if feh_bib is None:
                                feh_bib = val
                            if teff_bib is None:
                                teff_bib = val
                            if logg_bib is None:
                                logg_bib = val

        simbad_params = {
            "teff": teff,
            "logg": logg,
            "feh": feh,
            "parallax": parallax,
            "teff_ref": teff_bib,
            "logg_ref": logg_bib,
            "feh_ref": feh_bib,
            "parallax_ref": parallax_bib,
            "teff_unit": teff_unit,
            "logg_unit": logg_unit,
            "feh_unit": feh_unit,
            "parallax_unit": parallax_unit,
            "source": "SIMBAD",
        }

        if any(v is not None for v in (teff, logg, feh, parallax)):
            log.info("SIMBAD query successful: Teff=%s, logg=%s, feh=%s, parallax=%s", teff, logg, feh, parallax)
            return simbad_params

        return None
    except Exception as exc:
        log.warning("SIMBAD query failed: %s", exc)
        return None


