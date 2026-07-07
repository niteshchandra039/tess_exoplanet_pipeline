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
    sectors: int | str | list[int] | None = None,
) -> Any:
    """
    Load one or more local FITS files into a LightCurveCollection.

    Parameters
    ----------
    fits_paths : str | Path | list[str | Path] | tuple[str | Path, ...]
        FITS file path(s). Directory paths are allowed and will load `*.fits*`.
    flux_column : str
        Flux column to extract when reading SPOC files (default: "pdcsap_flux").
    sectors : int | str | list[int] | None
        Sectors selection to filter the loaded FITS files.

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

    if sectors is not None:
        def get_sector_from_file_or_header(path: Path) -> int | None:
            import re
            match = re.search(r"-s(\d{4})-", path.name)
            if match:
                return int(match.group(1))
            try:
                from astropy.io import fits
                with fits.open(path) as hdul:
                    for hdu in hdul:
                        if "SECTOR" in hdu.header:
                            return int(hdu.header["SECTOR"])
            except Exception:
                pass
            return None

        file_sectors = []
        for f in unique_files:
            sec = get_sector_from_file_or_header(f)
            if sec is not None:
                file_sectors.append((f, sec))

        if file_sectors:
            file_sectors.sort(key=lambda x: x[1])
            unique_sectors = sorted(list(set(sec for _, sec in file_sectors)))
            num_sectors = len(unique_sectors)

            indices = []
            selected_sector_numbers = []
            if sectors == "all":
                indices = list(range(num_sectors))
                selected_sector_numbers = [unique_sectors[i] for i in indices]
            elif sectors == "longest":
                runs = []
                if unique_sectors:
                    current_run = [unique_sectors[0]]
                    for x in unique_sectors[1:]:
                        if x == current_run[-1] + 1:
                            current_run.append(x)
                        else:
                            runs.append(current_run)
                            current_run = [x]
                    runs.append(current_run)
                selected_sector_numbers = max(runs, key=len) if runs else []
            elif isinstance(sectors, int):
                n = min(sectors, num_sectors)
                indices = list(range(n))
                selected_sector_numbers = [unique_sectors[i] for i in indices]
            elif isinstance(sectors, str) and sectors.startswith("slice:"):
                slice_str = sectors.split(":", 1)[1]
                parts = slice_str.split(":")
                start_val = parts[0].strip()
                end_val = parts[1].strip()
                start_idx = int(start_val) if start_val else 1
                end_idx = int(end_val) if end_val else num_sectors
                for idx in range(start_idx - 1, end_idx):
                    if 0 <= idx < num_sectors:
                        indices.append(idx)
                selected_sector_numbers = [unique_sectors[i] for i in indices]
            elif isinstance(sectors, (list, tuple)):
                selected_sector_numbers = [sec for sec in unique_sectors if sec in [int(s) for s in sectors]]
            unique_files = [f for f, sec in file_sectors if sec in selected_sector_numbers]
            if not unique_files:
                raise DataDownloadError(
                    f"Sectors selection {sectors!r} matched no available local FITS files. "
                    f"Available sectors: {unique_sectors}"
                )
            log.info("[fits] Filtered FITS files to sector(s) %s", selected_sector_numbers)

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
    sectors: int | str | list[int] = 1,
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
    sectors : int | str | list[int]
        Number/selection of sectors to download.
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

    # Scan local FITS files first
    fits_dir = Path("data/fits")
    fits_dir.mkdir(parents=True, exist_ok=True)
    tic_str = f"{tic_id:016d}"
    existing_files = list(fits_dir.glob(f"*-{tic_id}-*.fits*")) + list(fits_dir.glob(f"*-{tic_str}-*.fits*"))
    existing_files = list(set(existing_files))  # deduplicate

    def get_sector_from_file_or_header(path: Path) -> int | None:
        import re
        match = re.search(r"-s(\d{4})-", path.name)
        if match:
            return int(match.group(1))
        try:
            from astropy.io import fits
            with fits.open(path) as hdul:
                for hdu in hdul:
                    if "SECTOR" in hdu.header:
                        return int(hdu.header["SECTOR"])
        except Exception:
            pass
        return None

    local_file_sectors = {}
    for f in existing_files:
        sec = get_sector_from_file_or_header(f)
        if sec is not None:
            if sec not in local_file_sectors:
                local_file_sectors[sec] = []
            local_file_sectors[sec].append(f)

    # Determine available sectors
    online = True
    available_sectors = []
    mast_search = None

    old_timeout = mast_conf.timeout
    mast_conf.timeout = request_timeout
    try:
        mast_search = lk.search_lightcurve(
            f"TIC {tic_id}",
            author=author,
            exptime=cadence,
            mission="TESS",
        )
        if len(mast_search) > 0:
            table = mast_search.table
            if "sequence_number" in table.colnames:
                available_sectors = sorted(list(set(int(x) for x in table["sequence_number"])))
            else:
                available_sectors = list(range(1, len(mast_search) + 1))
    except Exception as exc:
        if force_download or not existing_files:
            raise DataDownloadError(
                f"MAST search failed for TIC {tic_id}: {exc}. Check connection."
            ) from exc
        else:
            log.warning("[download] MAST search failed: %s. Using local files only.", exc)
            online = False
            available_sectors = sorted(list(local_file_sectors.keys()))
    finally:
        mast_conf.timeout = old_timeout

    if not available_sectors:
        raise NoCadenceDataError(
            f"No TESS data found for TIC {tic_id}."
        )

    log.info("[download] Found %d available sector(s): %s", len(available_sectors), available_sectors)

    # Filter sectors selection
    num_sectors = len(available_sectors)
    indices = []
    selected_sectors = []
    if sectors == "all":
        indices = list(range(num_sectors))
        selected_sectors = [available_sectors[i] for i in indices]
    elif sectors == "longest":
        runs = []
        if available_sectors:
            current_run = [available_sectors[0]]
            for x in available_sectors[1:]:
                if x == current_run[-1] + 1:
                    current_run.append(x)
                else:
                    runs.append(current_run)
                    current_run = [x]
            runs.append(current_run)
        selected_sectors = max(runs, key=len) if runs else []
    elif isinstance(sectors, int):
        n = min(sectors, num_sectors)
        indices = list(range(n))
        selected_sectors = [available_sectors[i] for i in indices]
    elif isinstance(sectors, str) and sectors.startswith("slice:"):
        slice_str = sectors.split(":", 1)[1]
        parts = slice_str.split(":")
        start_val = parts[0].strip()
        end_val = parts[1].strip()
        start_idx = int(start_val) if start_val else 1
        end_idx = int(end_val) if end_val else num_sectors
        for idx in range(start_idx - 1, end_idx):
            if 0 <= idx < num_sectors:
                indices.append(idx)
        selected_sectors = [available_sectors[i] for i in indices]
    elif isinstance(sectors, (list, tuple)):
        selected_sectors = [sec for sec in available_sectors if sec in [int(s) for s in sectors]]
    if not selected_sectors:
        raise DataDownloadError(
            f"Sectors selection {sectors!r} matched no available sectors. Available: {available_sectors}"
        )

    log.info("[download] Selected sectors for analysis: %s", selected_sectors)

    # Check if we have all selected sectors locally
    have_all_local = True
    local_files_to_load = []
    for sec in selected_sectors:
        if sec in local_file_sectors:
            local_files_to_load.extend(local_file_sectors[sec])
        else:
            have_all_local = False
            break

    if not force_download and have_all_local and local_files_to_load:
        log.info("[download] Found local cached files for all selected sectors, loading them.")
        return load_lightcurves_from_fits(local_files_to_load)

    # Otherwise download from MAST
    if not online or mast_search is None:
        raise DataDownloadError(
            f"Offline but missing cached files for selected sectors {selected_sectors}. "
            f"Available locally: {list(local_file_sectors.keys())}"
        )

    table = mast_search.table
    selected_indices = []
    for idx in range(len(mast_search)):
        sec = int(table["sequence_number"][idx]) if "sequence_number" in table.colnames else (idx + 1)
        if sec in selected_sectors:
            selected_indices.append(idx)

    selected_search = mast_search[selected_indices]
    log.info("[download] Downloading %d selected sector(s) from MAST", len(selected_search))

    import shutil
    try:
        lc_collection = selected_search.download_all(
            flux_column="pdcsap_flux",
            download_dir="data/fits",
            quality_bitmask="hardest",
        )
        
        # Flatten downloaded files to data/fits/
        mast_dir = Path("data/fits/mastDownload")
        if mast_dir.exists():
            for fits_file in mast_dir.glob("**/*.fits*"):
                dest = Path("data/fits") / fits_file.name
                shutil.move(str(fits_file), str(dest))
            shutil.rmtree(str(mast_dir))
    except Exception as exc:
        raise DataDownloadError(
            f"Download failed for TIC {tic_id}: {exc}."
        ) from exc

    # Re-scan local files after download to load the correct filtered set
    existing_files_after = list(fits_dir.glob(f"*-{tic_id}-*.fits*")) + list(fits_dir.glob(f"*-{tic_str}-*.fits*"))
    existing_files_after = list(set(existing_files_after))

    local_file_sectors_after = {}
    for f in existing_files_after:
        sec = get_sector_from_file_or_header(f)
        if sec is not None:
            if sec not in local_file_sectors_after:
                local_file_sectors_after[sec] = []
            local_file_sectors_after[sec].append(f)

    files_to_load = []
    for sec in selected_sectors:
        if sec in local_file_sectors_after:
            files_to_load.extend(local_file_sectors_after[sec])

    lc_collection = load_lightcurves_from_fits(files_to_load)
    if lc_collection is None or len(lc_collection) == 0:
        raise DataDownloadError(
            f"Load returned empty collection for TIC {tic_id} after download."
        )

    elapsed = time.perf_counter() - start_time
    log.info(
        "[download] Completed TIC %d download in %.1fs (%d sector file(s))",
        tic_id,
        elapsed,
        len(lc_collection),
    )

    return lc_collection
