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
                "epoch": archive_result.get("epoch"),
            }
            log.info(
                "Archive period: %.6f d (ref: %s)",
                self.archive_period,
                archive_result.get("reference", ""),
            )
        return self.archive_period

    def search(self) -> dict[str, Any]:
        """Run TLS/BLS period search (support multiple planets)."""
        cfg = self.config
        lc = self.results.lightcurve
        if lc is None:
            raise RuntimeError("Call load_lightcurves() and preprocess() before search_period()")

        from tess_pipeline.transit.detection import search_multiple_planets, search_period
        import numpy as np

        detections = []

        if self.archive_period is not None:
            is_override = (self.results.period.get("source") == "override")
            log.info("%s period found: %.6f d; refining Planet 1", "Override" if is_override else "Archive", self.archive_period)
            try:
                half_width = min(0.01, self.archive_period * 0.002)
                search_min = max(cfg.period_min, self.archive_period - half_width)
                search_max = min(cfg.period_max, self.archive_period + half_width)

                det1 = search_period(
                    lc,
                    method=cfg.search_method,
                    period_min=search_min,
                    period_max=search_max,
                    stellar=None,
                    is_archive=True,
                )
                if is_override:
                    det1["period"] = self.archive_period
                    det1["method"] = "override"
                    det1["note"] = "user override preserved"
                else:
                    det1["note"] = "refined archive period"
            except Exception as exc:
                log.warning("Refinement of archive/override period failed: %s", exc)
                det1 = {
                    "period": self.archive_period,
                    "epoch": float(lc.time.value[np.argmin(lc.flux.value)]),
                    "duration_hr": 3.0,
                    "depth": float(1.0 - np.percentile(lc.flux.value, 1)),
                    "method": "override" if is_override else "archive",
                    "sde": 10.0,
                    "snr": 10.0,
                    "note": "override fallback" if is_override else "archive fallback",
                }
            detections.append(det1)
            
            # If max_planets > 1, search for more planets on the masked lightcurve
            if cfg.max_planets > 1:
                period = det1["period"]
                t0 = det1["epoch"]
                duration_days = (det1["duration_hr"] or 3.0) / 24.0
                times = lc.time.value
                phase = (times - t0 + 0.5 * period) % period - 0.5 * period
                in_transit = np.abs(phase) < (0.75 * duration_days)
                
                masked_lc = lc[~in_transit]
                
                extra_dets = search_multiple_planets(
                    masked_lc,
                    method=cfg.search_method,
                    period_min=cfg.period_min,
                    period_max=cfg.period_max,
                    stellar=None,
                    max_planets=cfg.max_planets - 1,
                )
                detections.extend(extra_dets)
        else:
            # No archive period, search all planets from scratch
            detections = search_multiple_planets(
                lc,
                method=cfg.search_method,
                period_min=cfg.period_min,
                period_max=cfg.period_max,
                stellar=None,
                max_planets=cfg.max_planets,
            )

        # Store detections in results
        self.results.metadata["detections"] = detections
        if detections:
            self.results.detection = detections[0]
            self.results.period = {
                "value": detections[0]["period"],
                "source": detections[0]["method"],
            }
            if self.archive_period is not None:
                self.results.period["source"] = "override" if is_override else "archive"
            log.info("Detections completed. Found %d planet candidate(s).", len(detections))
        else:
            raise RuntimeError("No period search detections were found.")

        return self.results.detection

    @property
    def best_period(self) -> float:
        if not self.results.period:
            raise RuntimeError("No period available; run lookup_archive() or search() first")
        return self.results.period["value"]
