"""Archive period lookup and TLS/BLS period search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tess_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from tess_pipeline.config import PipelineConfig
    from tess_pipeline.results import PipelineResults

log = get_logger(__name__)


class PeriodStage:
    """Look up a known period and/or run TLS/BLS detection."""

    def __init__(self, config: PipelineConfig, results: PipelineResults) -> None:
        self.config = config
        self.results = results
        self.archive_period: float | None = None

    def lookup_archive(self) -> float | None:
        """Query NASA Exoplanet Archive or use a configured period override."""
        cfg = self.config
        tic_id = self.results.target["tic_id"]

        if cfg.period_override is not None:
            log.info("Using overridden period: %.6f d", cfg.period_override)
            self.archive_period = cfg.period_override
            self.results.period = {"value": self.archive_period, "source": "override"}
            return self.archive_period

        log.info("Querying NASA Exoplanet Archive for TIC %s", tic_id)
        from tess_pipeline.catalogs.nasa_archive import query_archive

        archive_result = query_archive(tic_id)
        if archive_result.get("period") is not None:
            self.archive_period = archive_result["period"]
            self.results.period = {
                "value": self.archive_period,
                "source": "archive",
                "reference": archive_result.get("reference", ""),
            }
            log.info(
                "Archive period: %.6f d (ref: %s)",
                self.archive_period,
                archive_result.get("reference", ""),
            )
        return self.archive_period

    def search(self) -> dict[str, Any]:
        """Run TLS/BLS when no archive period is available (diagnostic if archive exists)."""
        cfg = self.config
        lc = self.results.lightcurve
        if lc is None:
            raise RuntimeError("Call load_lightcurves() and preprocess() before search_period()")

        if self.archive_period is None:
            log.info("No archive period; running %s search", cfg.search_method.upper())
            from tess_pipeline.transit.detection import search_period

            detection = search_period(
                lc,
                method=cfg.search_method,
                period_min=cfg.period_min,
                period_max=cfg.period_max,
                stellar=None,
            )
            self.results.detection = detection
            self.results.period = {
                "value": detection["period"],
                "source": detection["method"],
            }
            log.info(
                "Best period: %.6f d (SDE/SNR = %.2f)",
                detection["period"],
                detection.get("sde", detection.get("snr", float("nan"))),
            )
            return detection

        try:
            from tess_pipeline.transit.detection import search_period

            detection = search_period(
                lc,
                method=cfg.search_method,
                period_min=cfg.period_min,
                period_max=cfg.period_max,
                stellar=None,
            )
            detection["note"] = "diagnostic only; archive period used"
            self.results.detection = detection
        except Exception as exc:  # noqa: BLE001
            log.warning("Period search for diagnostics failed: %s", exc)
            self.results.detection = {
                "period": self.archive_period,
                "epoch": None,
                "duration_hr": None,
                "depth": None,
                "note": "archive period; no TLS/BLS run",
            }
        return self.results.detection

    @property
    def best_period(self) -> float:
        if not self.results.period:
            raise RuntimeError("No period available; run lookup_archive() or search() first")
        return self.results.period["value"]
