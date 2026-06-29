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
        """Query Gaia DR3 for the target."""
        target_info = self.results.target
        tic_id = target_info["tic_id"]

        log.info("Querying Gaia DR3")
        from tess_pipeline.catalogs.gaia import query_gaia

        self.gaia_params = query_gaia(
            ra=target_info.get("ra"), dec=target_info.get("dec"), tic_id=tic_id
        )
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
        """Query SDSS for radial velocity (may return None)."""
        target_info = self.results.target

        log.info("Querying SDSS for radial velocity")
        from tess_pipeline.catalogs.sdss import query_sdss_rv

        rv = query_sdss_rv(ra=target_info.get("ra"), dec=target_info.get("dec"))
        self.results.rv = rv
        if rv is None:
            log.info("No SDSS RV available for this target")
        return rv
