"""Download or load TESS light curves and preprocess them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.config import PipelineConfig
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


class LightCurveStage:
    """Load raw light curves and preprocess into a single stitched curve."""

    def __init__(self, config: PipelineConfig, results: PipelineResults) -> None:
        self.config = config
        self.results = results
        self.raw_collection: Any = None

    def load(self) -> Any:
        """Download from MAST or load local FITS files."""
        cfg = self.config
        tic_id = self.results.target["tic_id"]

        if cfg.lightcurve_source == "fits":
            log.info("Loading TESS light curves from local FITS files")
            from tess_pipeline.data.download import load_lightcurves_from_fits

            self.raw_collection = load_lightcurves_from_fits(cfg.lightcurve_fits)
        else:
            log.info("Downloading TESS data for TIC %s", tic_id)
            from tess_pipeline.data.download import download_lightcurves

            self.raw_collection = download_lightcurves(
                tic_id,
                author=cfg.author,
                cadence=cfg.cadence,
                sectors=cfg.sectors,
                force_download=cfg.force_download,
            )

        return self.raw_collection

    def preprocess(self) -> Any:
        """Sigma-clip, stitch, and flatten the loaded light curves."""
        if self.raw_collection is None:
            raise RuntimeError("Call load() before preprocess()")

        cfg = self.config
        log.info("Preprocessing light curves")
        from tess_pipeline.data.preprocess import preprocess

        lc = preprocess(
            self.raw_collection,
            sigma_clip_lower=cfg.sigma_clip_lower,
            sigma_clip_upper=cfg.sigma_clip_upper,
            flatten_window_length=cfg.flatten_window_length,
            flatten_polyorder=cfg.flatten_polyorder,
            flatten_break_tolerance=cfg.flatten_break_tolerance,
        )
        self.results.lightcurve = lc
        return lc
