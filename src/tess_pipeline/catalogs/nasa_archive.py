"""
catalogs/nasa_archive.py — Read a local NASA Exoplanet Archive export for
published orbital periods and planet parameters for a given TIC ID.

Returns None for period when no catalog entry is found; never fabricates values.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)


def query_archive_all(tic_id: int) -> list[dict[str, Any]]:
    """
    Search the NASA Exoplanet Archive for *tic_id* and return all planets.
    """
    archive_path = _find_local_archive_csv()
    if archive_path is None:
        log.info("No local NASA archive CSV found; skipping archive lookup")
        return []

    rows = _load_local_archive_rows(archive_path)
    matches = [row for row in rows if _row_tic_id(row) == tic_id]
    
    # Filter only entries that are CP, KP, PC (optional based on preference, but usually good to keep all valid ones)
    # We will just parse all matching ones with valid periods and group by TOI id to avoid duplicates
    results = []
    seen_tois = set()
    
    # Sort them by rank so we process the best variants first
    matches.sort(key=_record_sort_key)
    
    for row in matches:
        toi = row.get("toi", "")
        if toi in seen_tois and toi:
            continue
            
        period = _safe_float(row.get("pl_orbper"))
        if period is None:
            continue
            
        if toi:
            seen_tois.add(toi)
            
        results.append({
            "period": period,
            "epoch": _safe_float(row.get("pl_tranmid")),
            "planet_name": f"TOI-{toi}" if toi else None,
            "rp_earth": _safe_float(row.get("pl_rade")),
            "t_eq": _safe_float(row.get("pl_eqt")),
            "reference": f"Local NASA Exoplanet Archive TOI export: {archive_path}",
            "source": "toi-local"
        })
        
    if not results:
        log.info("No valid periods found in local NASA archive entry for TIC %d", tic_id)
        
    return results

def query_archive(tic_id: int) -> dict[str, Any]:
    """
    Search the NASA Exoplanet Archive for *tic_id*.

    Returns
    -------
    dict with keys:
        period (float | None)   : orbital period in days
        epoch (float | None)    : transit mid-time (BJD)
        planet_name (str | None): planet designation (e.g. "TOI-270 b")
        rp_earth (float | None) : published planet radius in Earth radii
        t_eq (float | None)     : published equilibrium temperature (K)
        reference (str)         : ADS bibcode or journal reference
        source (str)            : "confirmed" | "toi" | "none"
    """
    result = _query_local_toi(tic_id)
    if result.get("period") is not None:
        return result

    log.info("No local NASA archive entry found for TIC %d", tic_id)
    return {
        "period": None,
        "epoch": None,
        "planet_name": None,
        "rp_earth": None,
        "t_eq": None,
        "reference": "",
        "source": "none",
    }


def get_local_archive_record(tic_id: int) -> dict[str, str] | None:
    """Return the best matching local TOI row for a TIC ID, if present."""
    archive_path = _find_local_archive_csv()
    if archive_path is None:
        log.info("No local NASA archive CSV found; skipping archive lookup")
        return None

    rows = _load_local_archive_rows(archive_path)
    matches = [row for row in rows if _row_tic_id(row) == tic_id]
    if not matches:
        return None

    matches.sort(key=_record_sort_key)
    best = matches[0]
    log.info(
        "Using local NASA archive CSV %s for TIC %d (matched %d row(s))",
        archive_path,
        tic_id,
        len(matches),
    )
    return best


def _query_local_toi(tic_id: int) -> dict[str, Any]:
    """Query the local TOI CSV export for a TIC ID."""
    row = get_local_archive_record(tic_id)
    if row is None:
        return {"period": None, "source": "none"}

    period = _safe_float(row.get("pl_orbper"))
    if period is None:
        return {"period": None, "source": "none"}

    toi = row.get("toi", "")
    planet_name = f"TOI-{toi}" if toi else None
    return {
        "period": period,
        "epoch": _safe_float(row.get("pl_tranmid")),
        "planet_name": planet_name,
        "rp_earth": _safe_float(row.get("pl_rade")),
        "t_eq": _safe_float(row.get("pl_eqt")),
        "reference": f"Local NASA Exoplanet Archive TOI export: {_find_local_archive_csv()}",
        "source": "toi-local",
    }


def _find_local_archive_csv() -> Path | None:
    """Locate the local TOI CSV export, preferring explicit env override."""
    env_path = os.environ.get("TESS_PIPELINE_ARCHIVE_CSV")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.extend(sorted((parent / "data").glob("TOI_*.csv"), reverse=True))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


@lru_cache(maxsize=4)
def _load_local_archive_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Load and cache the local TOI CSV rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        filtered = (line for line in handle if not line.startswith("#"))
        reader = csv.DictReader(filtered)
        return tuple(dict(row) for row in reader)


def _row_tic_id(row: dict[str, str]) -> int | None:
    """Extract TIC ID from a local archive row."""
    for key in ("tid", "ticid", "tic_id"):
        value = row.get(key)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _record_sort_key(row: dict[str, str]) -> tuple[int, int, float]:
    """Prefer confirmed/planet-candidate rows and lower pipeline numbers."""
    disposition = (row.get("tfopwg_disp") or "").strip().upper()
    disposition_rank = {
        "CP": 0,
        "KP": 1,
        "PC": 2,
        "APC": 3,
        "FA": 4,
        "FP": 5,
    }.get(disposition, 99)
    signal_id = int(_safe_float(row.get("pl_pnum")) or 9999)
    toi = float(_safe_float(row.get("toi")) or 999999.0)
    return (disposition_rank, signal_id, toi)


def _safe_float(value: Any) -> float | None:
    """Return float if *value* is a finite number, else None."""
    import math

    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
