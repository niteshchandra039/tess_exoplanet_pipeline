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
        self.archive_periods: list[dict[str, Any]] = []

    def lookup_archive(self) -> float | None:
        """Query NASA Exoplanet Archive or use a configured period override."""
        cfg = self.config
        tic_id = self.results.target["tic_id"]

        if cfg.period_override is not None:
            log.info("Using overridden period: %.6f d", cfg.period_override)
            self.archive_period = cfg.period_override
            self.archive_periods = [{
                "period": cfg.period_override,
                "epoch": None,
                "source": "override"
            }]
            self.results.period = {"value": self.archive_period, "source": "override"}
            return self.archive_period

        log.info("Querying NASA Exoplanet Archive for TIC %s", tic_id)
        from tess_pipeline.catalogs.nasa_archive import query_archive_all

        archive_results = query_archive_all(tic_id)
        if archive_results:
            self.archive_periods = archive_results
            self.archive_period = archive_results[0]["period"]
            self.results.period = {
                "value": self.archive_period,
                "source": "archive",
                "reference": archive_results[0].get("reference", ""),
                "epoch": archive_results[0].get("epoch"),
                "all_periods": [r["period"] for r in archive_results]
            }
            log.info(
                "Archive periods found: %d planets (Primary: %.6f d)",
                len(archive_results),
                self.archive_period,
            )
            
            # Dynamically adjust max_planets if it's smaller than the archive list
            from tess_pipeline.constants import DEFAULT_MAX_PLANETS
            if cfg.max_planets == DEFAULT_MAX_PLANETS and len(archive_results) > DEFAULT_MAX_PLANETS:
                log.info("No max_planets provided; defaulting to number of literature entries (%d)", len(archive_results))
                cfg.max_planets = len(archive_results)
            elif cfg.max_planets < len(archive_results):
                log.info("Increasing max_planets from %d to %d to accommodate all literature entries", cfg.max_planets, len(archive_results))
                cfg.max_planets = len(archive_results)
                
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

        if self.archive_periods:
            is_override = (self.results.period.get("source") == "override")
            masked_lc = lc.copy()
            
            for i, arch in enumerate(self.archive_periods):
                p_arch = arch["period"]
                log.info("%s period found for Planet %d: %.6f d; refining...", 
                         "Override" if is_override else "Archive", i + 1, p_arch)
                try:
                    half_width = min(0.01, p_arch * 0.002)
                    search_min = max(cfg.period_min, p_arch - half_width)
                    search_max = min(cfg.period_max, p_arch + half_width)

                    det = search_period(
                        masked_lc,
                        method=cfg.search_method,
                        period_min=search_min,
                        period_max=search_max,
                        stellar=None,
                        is_archive=True,
                    )
                    if is_override:
                        det["period"] = p_arch
                        det["method"] = "override"
                        det["note"] = "user override preserved"
                    else:
                        det["note"] = "refined archive period"
                except Exception as exc:
                    log.warning("Refinement of archive/override period failed: %s", exc)
                    det = {
                        "period": p_arch,
                        "epoch": arch.get("epoch") or float(masked_lc.time.value[np.argmin(masked_lc.flux.value)]),
                        "duration_hr": 3.0,
                        "depth": float(1.0 - np.percentile(masked_lc.flux.value, 1)),
                        "method": "override" if is_override else "archive",
                        "sde": 10.0,
                        "snr": 10.0,
                        "note": "override fallback" if is_override else "archive fallback",
                    }
                detections.append(det)
                
                # Mask this planet specifically
                period = det["period"]
                t0 = det["epoch"]
                duration_days = (det["duration_hr"] or 3.0) / 24.0
                times = masked_lc.time.value
                phase = (times - t0 + 0.5 * period) % period - 0.5 * period
                in_transit = np.abs(phase) < (0.75 * duration_days)
                
                masked_lc = masked_lc[~in_transit]
                
            # Now search for any extra planets using the fully masked lightcurve
            remaining_planets_to_find = cfg.max_planets - len(self.archive_periods)
            if remaining_planets_to_find > 0:
                log.info("Searching for %d additional planet(s) in residuals...", remaining_planets_to_find)
                extra_dets = search_multiple_planets(
                    masked_lc,
                    method=cfg.search_method,
                    period_min=cfg.period_min,
                    period_max=cfg.period_max,
                    stellar=None,
                    max_planets=remaining_planets_to_find,
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
