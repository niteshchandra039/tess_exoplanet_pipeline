"""Resolve a TIC identifier and target metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.config import PipelineConfig
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


class TargetStage:
    """Resolve the pipeline target to a TIC ID and coordinates."""

    def __init__(self, config: PipelineConfig, results: PipelineResults) -> None:
        self.config = config
        self.results = results

    def run(self) -> dict[str, Any]:
        from tess_pipeline.data.metadata import resolve_target, resolve_target_from_fits

        cfg = self.config

        if cfg.lightcurve_source == "fits":
            log.info("Resolving target from local FITS inputs")
            target_info = resolve_target_from_fits(
                cfg.lightcurve_fits,
                target_fallback=cfg.target or None,
                allow_remote=False,
            )
        else:
            log.info("Resolving target: %s", cfg.target)
            target_info = resolve_target(cfg.target)

        self.results.target = target_info
        log.info("Resolved to TIC %s", target_info["tic_id"])
        return target_info
