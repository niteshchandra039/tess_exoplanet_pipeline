"""Gaia DR3, stellar characterization, and optional SDSS RV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.config import PipelineConfig
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


class StellarStage:
    """Query external catalogs and derive stellar parameters."""

    def __init__(self, config: PipelineConfig, results: PipelineResults) -> None:
        self.config = config
        self.results = results
        self.gaia_params: dict[str, Any] = {}

    def query_gaia(self) -> dict[str, Any]:
        """Query Gaia DR3 for the target and supplement with archive/TIC if needed."""
        target_info = self.results.target
        tic_id = target_info["tic_id"]

        log.info("Querying Gaia DR3")
        from tess_pipeline.catalogs.gaia import query_gaia

        self.gaia_params = query_gaia(
            ra=target_info.get("ra"), dec=target_info.get("dec"), tic_id=tic_id
        )
        self.gaia_params["tic_id"] = tic_id

        # 1. Supplement missing parameters from local NASA archive
        missing_keys = [k for k in ["r_star", "teff", "logg"] if self.gaia_params.get(k) is None]
        if missing_keys:
            log.info("Parameters %s missing in Gaia DR3; checking local NASA TOI archive", missing_keys)
            from tess_pipeline.catalogs.nasa_archive import get_local_archive_record
            row = get_local_archive_record(tic_id)
            if row is not None:
                mapping = {
                    "r_star": "st_rad",
                    "r_star_err": "st_raderr1",
                    "teff": "st_teff",
                    "teff_err": "st_tefferr1",
                    "logg": "st_logg",
                    "logg_err": "st_loggerr1",
                }
                for key in missing_keys:
                    col = mapping.get(key)
                    val = row.get(col) if col else None
                    if val not in (None, "", "nan", "NaN"):
                        try:
                            self.gaia_params[key] = float(val)
                            err_key = f"{key}_err"
                            err_col = mapping.get(err_key)
                            err_val = row.get(err_col) if err_col else None
                            if err_val not in (None, "", "nan", "NaN"):
                                self.gaia_params[err_key] = abs(float(err_val))
                            log.info("Retrieved %s = %s from local NASA archive", key, val)
                        except (TypeError, ValueError):
                            pass

        # 2. If still missing, query remote MAST TIC catalog
        missing_keys = [k for k in ["r_star", "teff", "logg"] if self.gaia_params.get(k) is None]
        if missing_keys:
            log.info("Parameters %s still missing; querying MAST TESS Input Catalog (TIC)", missing_keys)
            try:
                from astroquery.mast import Catalogs
                import numpy as np
                import math

                result = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
                if len(result) > 0:
                    row = result[0]
                    mapping = {
                        "r_star": ("rad", "e_rad"),
                        "teff": ("Teff", "e_Teff"),
                        "logg": ("logg", "e_logg"),
                    }
                    for key in missing_keys:
                        col_val, col_err = mapping[key]
                        if col_val in result.colnames:
                            val = row[col_val]
                            if val is not None and not (isinstance(val, (float, np.float64)) and math.isnan(val)):
                                self.gaia_params[key] = float(val)
                                log.info("Retrieved %s = %s from MAST TIC", key, val)
                                if col_err in result.colnames:
                                    err_val = row[col_err]
                                    if err_val is not None and not (isinstance(err_val, (float, np.float64)) and math.isnan(err_val)):
                                        self.gaia_params[f"{key}_err"] = float(err_val)
            except Exception as exc:
                log.warning("MAST TIC catalog query failed: %s", exc)

        return self.gaia_params

    def characterize(self) -> dict[str, Any]:
        """Derive Teff, logg, R★, ρ★ from Gaia photometry."""
        if not self.gaia_params:
            self.query_gaia()

        log.info("Running stellar characterization (%s)", self.config.stellar_method)
        from tess_pipeline.catalogs.stellar import characterize_star

        stellar = characterize_star(self.gaia_params, method=self.config.stellar_method)
        self.results.stellar = stellar
        return stellar

    def query_sdss(self) -> dict[str, Any] | None:
        """Query SDSS for radial velocity (disabled)."""
        log.info("SDSS checking is disabled in this pipeline version")
        self.results.rv = None
        return None

