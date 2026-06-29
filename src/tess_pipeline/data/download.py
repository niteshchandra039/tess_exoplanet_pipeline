"""
data/download.py — Download TESS SPOC PDCSAP light curves via lightkurve.

Always uses:
  * author = "SPOC"
  * PDCSAP_FLUX
  * 2-minute cadence (120 s)
    * One sector by default (or all sectors if requested)

Results are cached locally at ~/.cache/tess-pipeline/.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

from tess_pipeline.data.metadata import expand_fits_paths
from tess_pipeline.exceptions import DataDownloadError, NoCadenceDataError
from tess_pipeline.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tess-pipeline"


def load_lightcurves_from_fits(
    fits_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    flux_column: str = "pdcsap_flux",
) -> Any:
    """
    Load one or more local FITS files into a LightCurveCollection.

    Parameters
    ----------
    fits_paths : str | Path | list[str | Path] | tuple[str | Path, ...]
        FITS file path(s). Directory paths are allowed and will load `*.fits*`.
    flux_column : str
        Flux column to extract when reading SPOC files (default: "pdcsap_flux").

    Returns
    -------
    lightkurve.LightCurveCollection
    """
    start_time = time.perf_counter()
    log.info("[fits] Step 1/4: normalizing FITS paths")

    unique_files = expand_fits_paths(fits_paths)
    if not unique_files:
        raise DataDownloadError("No FITS paths provided for local light-curve loading.")

    missing = [str(p) for p in unique_files if not p.exists()]
    if missing:
        raise DataDownloadError(
            "FITS file(s) not found: " + ", ".join(missing)
        )

    import lightkurve as lk

    log.info("[fits] Step 2/4: %d FITS file(s) queued", len(unique_files))

    curves = []
    for i, fits_file in enumerate(unique_files, start=1):
        log.info("[fits] Step 3/4: reading file %d/%d: %s", i, len(unique_files), fits_file)
        try:
            obj = lk.read(str(fits_file), quality_bitmask="hardest")
            if hasattr(obj, "to_lightcurve"):
                lc = obj.to_lightcurve(flux_column=flux_column)
            else:
                lc = obj
            curves.append(lc)
        except Exception as exc:
            raise DataDownloadError(
                f"Failed to read FITS light curve from {fits_file}: {exc}"
            ) from exc

    collection = lk.LightCurveCollection(curves)
    elapsed = time.perf_counter() - start_time
    log.info(
        "[fits] Step 4/4: loaded %d light curve(s) from local FITS in %.1fs",
        len(collection),
        elapsed,
    )
    return collection


def _cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    d = Path(os.environ.get("TESS_PIPELINE_CACHE_DIR", _DEFAULT_CACHE_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(tic_id: int, author: str, cadence: int, sectors: int | str) -> str:
    raw = f"{tic_id}_{author}_{cadence}_{sectors}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]  # noqa: S324 (non-cryptographic cache key)


def download_lightcurves(
    tic_id: int,
    *,
    author: str = "SPOC",
    cadence: int = 120,
    sectors: int | str = 1,
    all_sectors: bool | None = None,
    force_download: bool = False,
    request_timeout: int = 120,
) -> Any:
    """
    Download TESS SPOC PDCSAP light curves for *tic_id*.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog identifier.
    author : str
        MAST pipeline author filter (default: "SPOC").
    cadence : int
        Exposure time in seconds (default: 120).
    sectors : int | str
        Number of sectors to download: 1 (default), 2, 3, or "all".
    all_sectors : bool | None
        Deprecated compatibility flag. If True, behaves as sectors="all".
    force_download : bool
        Bypass local cache and re-download (default: False).
    request_timeout : int
        Timeout in seconds for MAST requests via astroquery/lightkurve.

    Returns
    -------
    lightkurve.LightCurveCollection

    Raises
    ------
    NoCadenceDataError
        If no SPOC short-cadence data exists for this target.
    DataDownloadError
        If the download fails for any other reason.
    """
    import lightkurve as lk
    from astroquery.mast import conf as mast_conf

    start_time = time.perf_counter()
    log.info(
        "[download] Starting download for TIC %d (author=%s, cadence=%d, sectors=%s, force_download=%s)",
        tic_id,
        author,
        cadence,
        sectors,
        force_download,
    )

    if all_sectors is True:
        warnings.warn(
            "all_sectors is deprecated; use sectors='all' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        sectors = "all"

    if sectors not in (1, 2, 3, "all"):
        raise DataDownloadError(
            f"sectors must be one of 1, 2, 3, or 'all'; got {sectors!r}"
        )

    log.info("[download] Step 1/5: validated input arguments")

    fits_dir = Path("data/fits")
    fits_dir.mkdir(parents=True, exist_ok=True)
    tic_str = f"{tic_id:016d}"
    existing_files = list(fits_dir.glob(f"*-{tic_id}-*.fits*")) + list(fits_dir.glob(f"*-{tic_str}-*.fits*"))
    existing_files = list(set(existing_files))  # deduplicate

    if not force_download and existing_files:
        log.info("[download] Found existing files for TIC %d in data/fits/, loading them:", tic_id)
        for f in existing_files:
            log.info("  %s", f.name)
        return load_lightcurves_from_fits(existing_files)

    log.info("[download] Local FITS files not found (or force_download=True), querying MAST")

    old_timeout = mast_conf.timeout
    mast_conf.timeout = request_timeout
    log.info(
        "[download] Step 3/5: searching MAST for TIC %d with timeout=%ss",
        tic_id,
        request_timeout,
    )

    try:
        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            author=author,
            exptime=cadence,
            mission="TESS",
        )
    except Exception as exc:
        raise DataDownloadError(
            f"MAST search failed for TIC {tic_id}: {exc}. "
            "Check network connectivity, proxy/firewall settings, and MAST service status."
        ) from exc
    finally:
        mast_conf.timeout = old_timeout

    if len(search) == 0:
        raise NoCadenceDataError(
            f"No {author} {cadence}-second TESS data found for TIC {tic_id}. "
            "Try a different cadence or author."
        )

    log.info("[download] Found %d available sector(s) for TIC %d", len(search), tic_id)

    # Select subset of sectors when requested
    if sectors == "all":
        selected = search
    else:
        n = min(int(sectors), len(search))
        table = search.table
        if "sequence_number" in table.colnames:
            sorted_idx = sorted(
                range(len(search)),
                key=lambda i: table["sequence_number"][i],
            )
            selected = search[sorted_idx[:n]]
        else:
            selected = search[:n]
        log.info("[download] Step 4/5: selected %d sector(s) for analysis", n)

    log.info("[download] Step 5/5: downloading selected light curve products")

    import shutil
    try:
        lc_collection = selected.download_all(
            flux_column="pdcsap_flux",
            download_dir="data/fits",
            quality_bitmask="hardest",
        )
        
        # Flatten downloaded files to data/fits/
        downloaded_paths = []
        mast_dir = Path("data/fits/mastDownload")
        if mast_dir.exists():
            for fits_file in mast_dir.glob("**/*.fits*"):
                dest = Path("data/fits") / fits_file.name
                shutil.move(str(fits_file), str(dest))
                downloaded_paths.append(dest)
            shutil.rmtree(str(mast_dir))
            
            # Re-load from the flattened paths in data/fits/
            if downloaded_paths:
                lc_collection = load_lightcurves_from_fits(downloaded_paths)
    except Exception as exc:
        raise DataDownloadError(
            f"Download failed for TIC {tic_id}: {exc}. "
            "This can happen due to unstable network/timeout or temporary MAST issues."
        ) from exc

    if lc_collection is None or len(lc_collection) == 0:
        raise DataDownloadError(
            f"Download returned empty collection for TIC {tic_id}."
        )

    elapsed = time.perf_counter() - start_time
    log.info(
        "[download] Completed TIC %d download in %.1fs (%d sector file(s))",
        tic_id,
        elapsed,
        len(lc_collection),
    )

    return lc_collection
