"""
data/metadata.py — Target resolution and observation metadata.

Resolves flexible TIC ID strings to canonical integer TIC IDs and
fetches observation metadata (available sectors, cadences) from MAST.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tess_pipeline.exceptions import TargetResolutionError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)

_TIC_PATTERN = re.compile(r"(?:tic\s*)?(\d+)", re.IGNORECASE)
_TIC_SEARCH_PATTERN = re.compile(r"(?:tic[\s_\-]*)?(\d{6,16})", re.IGNORECASE)

_RA_HEADER_KEYS = ("RA_OBJ", "RA_TARG", "RA", "S_RA", "OBJCTRA")
_DEC_HEADER_KEYS = ("DEC_OBJ", "DEC_TARG", "DEC", "S_DEC", "OBJCTDEC")


def expand_fits_paths(
    fits_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> list[Path]:
    """Expand FITS path(s) and directories into a de-duplicated file list."""
    raw_paths: list[Path]
    if isinstance(fits_paths, (str, Path)):
        raw_paths = [Path(fits_paths)]
    else:
        raw_paths = [Path(p) for p in fits_paths]

    expanded_files: list[Path] = []
    for raw in raw_paths:
        if raw.is_dir():
            expanded_files.extend(sorted(raw.glob("*.fits*")))
        else:
            expanded_files.append(raw)

    return list(dict.fromkeys(expanded_files))


def infer_tic_id_from_fits_paths(
    fits_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> int | None:
    """
    Infer TIC ID from local FITS file path(s) using header fields and filename.

    Returns
    -------
    int | None
        Inferred TIC ID if found, otherwise None.
    """
    for path in expand_fits_paths(fits_paths):
        tic = _infer_tic_from_fits_header(path)
        if tic is not None:
            log.info("Inferred TIC %d from FITS header: %s", tic, path)
            return tic

        tic = _infer_tic_from_filename(path.name)
        if tic is not None:
            log.info("Inferred TIC %d from FITS filename: %s", tic, path)
            return tic

    return None


def infer_coordinates_from_fits_paths(
    fits_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> tuple[float | None, float | None]:
    """
    Infer ICRS RA/Dec (degrees) from FITS header keywords.

    TESS SPOC light curves typically store coordinates in ``RA_OBJ`` / ``DEC_OBJ``.
    """
    for path in expand_fits_paths(fits_paths):
        ra, dec = _infer_coordinates_from_fits_header(path)
        if ra is not None and dec is not None:
            log.info("Inferred coordinates from FITS header: %s (RA=%.6f, Dec=%.6f)", path, ra, dec)
            return ra, dec
    return None, None


def _infer_coordinates_from_fits_header(path: Path) -> tuple[float | None, float | None]:
    """Read RA/Dec in degrees from common FITS header keywords."""
    try:
        from astropy.io import fits

        with fits.open(path) as hdul:
            for header in (h.header for h in hdul):
                ra = _read_header_coordinate(header, _RA_HEADER_KEYS)
                dec = _read_header_coordinate(header, _DEC_HEADER_KEYS)
                if ra is not None and dec is not None:
                    return ra, dec
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not parse FITS coordinates for %s: %s", path, exc)
    return None, None


def _read_header_coordinate(header: Any, keys: tuple[str, ...]) -> float | None:
    """Return a coordinate in degrees from the first matching header keyword."""
    for key in keys:
        if key not in header:
            continue
        value = header.get(key)
        if value in (None, ""):
            continue
        try:
            if key in ("OBJCTRA", "OBJCTDEC"):
                from astropy.coordinates import Angle

                unit = "hourangle" if key == "OBJCTRA" else "deg"
                return float(Angle(value, unit=unit).deg)
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def resolve_target_from_fits(
    fits_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
    target_fallback: str | int | None = None,
    *,
    allow_remote: bool = False,
) -> dict[str, Any]:
    """
    Resolve target metadata from local FITS file(s).

    TIC ID and coordinates are read from FITS headers when available.
    *target_fallback* is only used when the TIC ID cannot be inferred.
    """
    tic_id = infer_tic_id_from_fits_paths(fits_paths)
    ra, dec = infer_coordinates_from_fits_paths(fits_paths)

    if tic_id is None:
        if target_fallback is not None and str(target_fallback).strip():
            return resolve_target(target_fallback, allow_remote=allow_remote)
        raise TargetResolutionError(
            "Could not infer TIC ID from FITS files. "
            "Ensure headers contain TICID/TIC/OBJECT or pass target= explicitly."
        )

    if ra is None or dec is None:
        archive_ra, archive_dec = _lookup_local_coordinates(tic_id)
        ra = ra if ra is not None else archive_ra
        dec = dec if dec is not None else archive_dec

    if allow_remote and (ra is None or dec is None):
        remote_ra, remote_dec = _fetch_coordinates(tic_id)
        ra = ra if ra is not None else remote_ra
        dec = dec if dec is not None else remote_dec

    return {
        "tic_id": tic_id,
        "name": f"TIC {tic_id}",
        "ra": ra,
        "dec": dec,
    }


def _infer_tic_from_fits_header(path: Path) -> int | None:
    """Try reading TIC ID from common FITS header keywords."""
    try:
        from astropy.io import fits

        with fits.open(path) as hdul:
            headers = [h.header for h in hdul]
            for header in headers:
                for key in ("TICID", "TIC", "OBJECT", "TARGETID"):
                    value = header.get(key)
                    tic = _extract_tic_id(value)
                    if tic is not None:
                        return tic
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not parse FITS header for %s: %s", path, exc)
    return None


def _infer_tic_from_filename(filename: str) -> int | None:
    """Extract TIC-like numeric token from a FITS filename."""
    # SPOC filenames often embed a 16-digit zero-padded target ID.
    padded_match = re.search(r"-(\d{10,16})-", filename)
    if padded_match:
        return int(padded_match.group(1))
    return _extract_tic_id(filename)


def _extract_tic_id(value: Any) -> int | None:
    """Extract TIC ID from free-form value or return None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _TIC_SEARCH_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def resolve_target(target: str | int, *, allow_remote: bool = True) -> dict[str, Any]:
    """
    Normalize *target* to a canonical TIC ID dict.

    Accepts
    -------
    - "TIC 307210830"
    - "TIC307210830"
    - "307210830"
    - 307210830  (int)

    Returns
    -------
    dict with keys: tic_id (int), ra (float|None), dec (float|None),
    name (str)
    """
    raw = str(target).strip()
    match = _TIC_PATTERN.fullmatch(raw)
    if match is None:
        raise TargetResolutionError(
            f"Cannot parse {raw!r} as a TIC ID. "
            "Expected formats: 'TIC 307210830', 'TIC307210830', or an integer."
        )

    tic_id = int(match.group(1))
    log.debug("Resolved %r to TIC %d", raw, tic_id)

    # Prefer local archive coordinates; only use remote lookup when allowed.
    ra, dec = _lookup_local_coordinates(tic_id)
    if ra is None and dec is None and allow_remote:
        ra, dec = _fetch_coordinates(tic_id)

    return {
        "tic_id": tic_id,
        "name": f"TIC {tic_id}",
        "ra": ra,
        "dec": dec,
    }


def _lookup_local_coordinates(tic_id: int) -> tuple[float | None, float | None]:
    """Return (RA, Dec) from the local TOI archive export when available."""
    try:
        from tess_pipeline.catalogs.nasa_archive import get_local_archive_record

        row = get_local_archive_record(tic_id)
        if row is None:
            return None, None
        ra = _safe_float(row.get("ra"))
        dec = _safe_float(row.get("dec"))
        if ra is not None or dec is not None:
            log.info("Loaded coordinates for TIC %d from local archive", tic_id)
        return ra, dec
    except Exception as exc:  # noqa: BLE001
        log.debug("Local archive coordinate lookup failed for TIC %d: %s", tic_id, exc)
        return None, None


def _fetch_coordinates(tic_id: int) -> tuple[float | None, float | None]:
    """Return (RA, Dec) in degrees for *tic_id* from the TIC via MAST."""
    try:
        import lightkurve as lk

        results = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(results) == 0:
            log.warning("No MAST observations found for TIC %d; coordinates unavailable", tic_id)
            return None, None
        table = results.table
        ra = float(table["s_ra"][0]) if "s_ra" in table.colnames else None
        dec = float(table["s_dec"][0]) if "s_dec" in table.colnames else None
        return ra, dec
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch coordinates for TIC %d: %s", tic_id, exc)
        return None, None


def _safe_float(value: Any) -> float | None:
    """Return float if *value* is numeric, else None."""
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def get_observation_metadata(tic_id: int, author: str = "SPOC", cadence: int = 120) -> dict[str, Any]:
    """
    Return available observation metadata for *tic_id*.

    Returns
    -------
    dict with keys:
        sectors (list[int])    : available sector numbers
        has_data (bool)        : True if any SPOC 2-min data exists
        n_sectors (int)        : number of sectors with data
    """
    try:
        import lightkurve as lk

        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            author=author,
            exptime=cadence,
            mission="TESS",
        )
        if len(search) == 0:
            return {"sectors": [], "has_data": False, "n_sectors": 0}

        table = search.table
        sectors: list[int] = []
        if "sequence_number" in table.colnames:
            sectors = sorted(int(s) for s in table["sequence_number"] if s is not None)

        return {
            "sectors": sectors,
            "has_data": True,
            "n_sectors": len(sectors),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Observation metadata lookup failed for TIC %d: %s", tic_id, exc)
        return {"sectors": [], "has_data": False, "n_sectors": 0}
