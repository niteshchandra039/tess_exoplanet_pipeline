"""
config.py — Pipeline configuration loading and validation.

Priority order (highest to lowest):
  1. Explicit TESSAnalysis() arguments
  2. Environment variables  (TESS_PIPELINE_*)
  3. User config file       (~/.config/tess-pipeline/config.toml or ./tess-pipeline.toml)
  4. Built-in defaults      (constants.py)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tess_pipeline import constants
from tess_pipeline.exceptions import ConfigurationError

# Python 3.11+ has tomllib in stdlib; earlier versions need tomli (not tomllib)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None  # type: ignore[assignment]


_CONFIG_SEARCH_PATHS = [
    Path("tess-pipeline.toml"),
    Path.home() / ".config" / "tess-pipeline" / "config.toml",
]


def _load_toml_file(path: Path) -> dict[str, Any]:
    """Return parsed TOML dict or empty dict if file does not exist."""
    if not path.exists():
        return {}
    if tomllib is None:
        import warnings

        warnings.warn(
            f"Config file {path} found but tomllib/tomli is not available. "
            "Install tomli or use Python ≥ 3.11 to load it.",
            stacklevel=3,
        )
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_user_config() -> dict[str, Any]:
    """Load the first config file found in the search paths."""
    for path in _CONFIG_SEARCH_PATHS:
        cfg = _load_toml_file(path)
        if cfg:
            return cfg
    return {}


def _env(key: str, default: Any, cast: type = str) -> Any:
    """Read a TESS_PIPELINE_* environment variable with type casting."""
    raw = os.environ.get(f"TESS_PIPELINE_{key.upper()}")
    if raw is None:
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes")
    try:
        return cast(raw)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            f"Invalid value for TESS_PIPELINE_{key.upper()!r}: {raw!r}"
        ) from exc


def _parse_sectors(value: Any) -> int | str:
    """Parse sectors value; accepts 1, 2, 3, or 'all'."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "all":
            return "all"
        try:
            value = int(v)
        except ValueError as exc:
            raise ConfigurationError(
                f"sectors must be one of 1, 2, 3, or 'all'; got {value!r}"
            ) from exc

    if value in (1, 2, 3):
        return int(value)

    raise ConfigurationError(
        f"sectors must be one of 1, 2, 3, or 'all'; got {value!r}"
    )


def _parse_lightcurve_source(value: Any) -> str:
    """Parse light curve source mode; accepts 'download' or 'fits'."""
    if not isinstance(value, str):
        raise ConfigurationError(
            f"lightcurve_source must be 'download' or 'fits'; got {value!r}"
        )
    mode = value.strip().lower()
    if mode not in ("download", "fits"):
        raise ConfigurationError(
            f"lightcurve_source must be 'download' or 'fits'; got {value!r}"
        )
    return mode


def _parse_fits_paths(value: Any) -> tuple[Path, ...]:
    """Parse optional FITS path(s) from list/tuple/CSV string/single path."""
    if value is None:
        return ()

    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            return ()
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return tuple(Path(p) for p in parts)

    if isinstance(value, (list, tuple)):
        paths: list[Path] = []
        for item in value:
            if isinstance(item, (str, Path)):
                item_text = str(item).strip()
                if item_text:
                    paths.append(Path(item_text))
            else:
                raise ConfigurationError(
                    "lightcurve_fits must contain only path strings"
                )
        return tuple(paths)

    raise ConfigurationError(
        "lightcurve_fits must be a path string, comma-separated string, or list of paths"
    )


def _parse_float_array(value: Any) -> tuple[float, ...]:
    """Parse optional float array/list/tuple/numpy array."""
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    try:
        import numpy as np
        return tuple(float(x) for x in np.asarray(value).flatten())
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Could not parse array/list of floats: {value!r}") from exc


@dataclass
class PipelineConfig:
    """Resolved, validated pipeline configuration."""

    # Target
    target: str = ""

    # TESS data
    author: str = constants.DEFAULT_AUTHOR
    cadence: int = constants.DEFAULT_CADENCE
    sectors: int | str = 1
    quality_bitmask: str = constants.DEFAULT_QUALITY_BITMASK
    force_download: bool = False
    lightcurve_source: str = "download"
    lightcurve_fits: tuple[Path, ...] = field(default_factory=tuple)

    # Preprocessing
    sigma_clip_lower: float = constants.DEFAULT_SIGMA_CLIP_LOWER
    sigma_clip_upper: float = constants.DEFAULT_SIGMA_CLIP_UPPER
    flatten_window_length: int = constants.DEFAULT_FLATTEN_WINDOW_LENGTH
    flatten_polyorder: int = constants.DEFAULT_FLATTEN_POLYORDER
    flatten_break_tolerance: int = constants.DEFAULT_FLATTEN_BREAK_TOLERANCE

    # Period search
    search_method: str = constants.DEFAULT_SEARCH_METHOD
    period_min: float = constants.DEFAULT_PERIOD_MIN
    period_max: float = constants.DEFAULT_PERIOD_MAX
    period_override: float | None = None
    max_planets: int = constants.DEFAULT_MAX_PLANETS

    # Stellar characterization
    stellar_method: str = constants.DEFAULT_STELLAR_METHOD

    # Inference
    inference: bool = True
    inference_backend: str = constants.DEFAULT_INFERENCE_BACKEND
    chains: int = constants.DEFAULT_CHAINS
    draws: int = constants.DEFAULT_DRAWS
    tune: int = constants.DEFAULT_TUNE
    target_accept: float = constants.DEFAULT_TARGET_ACCEPT
    gp_kernel: str = constants.DEFAULT_GP_KERNEL

    # Radial velocity data
    rv_times: tuple[float, ...] = field(default_factory=tuple)
    rv_vals: tuple[float, ...] = field(default_factory=tuple)
    rv_errs: tuple[float, ...] = field(default_factory=tuple)
    rv_file: str | None = None

    # Photometry configuration
    input_is_magnitude: bool = False

    # Output
    output_dir: Path = field(default_factory=lambda: Path(constants.DEFAULT_OUTPUT_DIR))
    plots: bool = True
    save_report: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.lightcurve_fits = tuple(Path(p) for p in self.lightcurve_fits)
        self._load_rv_file()
        self._validate()

    def _load_rv_file(self) -> None:
        if self.rv_file:
            path = Path(self.rv_file)
            if path.exists():
                times, vals, errs = [], [], []
                import csv
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read(1024)
                    f.seek(0)
                    delimiter = "," if "," in content else None
                    reader = csv.reader(f, delimiter=delimiter) if delimiter else csv.reader(f)
                    
                    for row in reader:
                        row = [x.strip() for x in row if x.strip()]
                        if not row or row[0].startswith("#"):
                            continue
                        try:
                            t = float(row[0])
                            v = float(row[1])
                            e = float(row[2]) if len(row) > 2 else 0.0
                            times.append(t)
                            vals.append(v)
                            errs.append(e)
                        except ValueError:
                            continue
                self.rv_times = tuple(times)
                self.rv_vals = tuple(vals)
                self.rv_errs = tuple(errs)
            else:
                raise ConfigurationError(f"Radial velocity file {self.rv_file} does not exist.")

    def _validate(self) -> None:
        if self.cadence not in (20, 120, 600, 1800):
            raise ConfigurationError(
                f"cadence must be 20, 120, 600, or 1800 seconds; got {self.cadence}"
            )
        if self.search_method not in ("tls", "bls", "both"):
            raise ConfigurationError(
                f"search_method must be 'tls', 'bls', or 'both'; got {self.search_method!r}"
            )
        if self.inference_backend not in ("exoplanet", "batman_only", "phoebe"):
            raise ConfigurationError(
                f"inference_backend must be 'exoplanet', 'batman_only', or 'phoebe'; "
                f"got {self.inference_backend!r}"
            )
        if self.chains < 1:
            raise ConfigurationError(f"chains must be ≥ 1; got {self.chains}")
        if self.draws < 1:
            raise ConfigurationError(f"draws must be ≥ 1; got {self.draws}")
        if self.sectors not in (1, 2, 3, "all"):
            raise ConfigurationError(
                f"sectors must be one of 1, 2, 3, or 'all'; got {self.sectors!r}"
            )
        if self.lightcurve_source not in ("download", "fits"):
            raise ConfigurationError(
                f"lightcurve_source must be 'download' or 'fits'; got {self.lightcurve_source!r}"
            )
        if self.lightcurve_source == "fits" and not self.lightcurve_fits:
            raise ConfigurationError(
                "lightcurve_fits is required when lightcurve_source='fits'"
            )
        if self.lightcurve_source == "download" and not self.target.strip():
            raise ConfigurationError(
                "target is required when lightcurve_source='download'"
            )
        if self.max_planets < 1:
            raise ConfigurationError(f"max_planets must be ≥ 1; got {self.max_planets}")


def build_config(
    target: str | int | None = None,
    *,
    author: str | None = None,
    cadence: int | None = None,
    sectors: int | str | None = None,
    all_sectors: bool | None = None,
    force_download: bool = False,
    lightcurve_source: str | None = None,
    lightcurve_fits: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    search_method: str | None = None,
    period_min: float | None = None,
    period_max: float | None = None,
    period_override: float | None = None,
    max_planets: int | None = None,
    stellar_method: str | None = None,
    inference: bool | None = None,
    inference_backend: str | None = None,
    chains: int | None = None,
    draws: int | None = None,
    tune: int | None = None,
    target_accept: float | None = None,
    gp_kernel: str | None = None,
    output_dir: str | Path | None = None,
    plots: bool | None = None,
    save_report: bool = False,
    verbose: bool | None = None,
    rv_times: Any = None,
    rv_vals: Any = None,
    rv_errs: Any = None,
    rv_file: str | Path | None = None,
    input_is_magnitude: bool | None = None,
) -> PipelineConfig:
    """Build a resolved PipelineConfig from all priority layers."""
    file_cfg = _load_user_config()
    fc = file_cfg  # shorthand

    def resolve(key: str, explicit: Any, cast: type, default: Any) -> Any:
        if explicit is not None:
            return explicit
        env_val = _env(key, None, cast)
        if env_val is not None:
            return env_val
        if key in fc:
            return cast(fc[key])
        return default

    # Backward compatibility with legacy all_sectors bool
    if sectors is None and all_sectors is True:
        sectors = "all"
    elif sectors is None and all_sectors is False:
        sectors = 1

    resolved_sectors = sectors
    if resolved_sectors is None:
        env_sectors = _env("sectors", None, str)
        if env_sectors is not None:
            resolved_sectors = env_sectors
        elif "sectors" in fc:
            resolved_sectors = fc["sectors"]
        elif "all_sectors" in fc:
            resolved_sectors = "all" if bool(fc["all_sectors"]) else 1
        else:
            resolved_sectors = 1

    resolved_sectors = _parse_sectors(resolved_sectors)

    resolved_lightcurve_source = lightcurve_source
    if resolved_lightcurve_source is None:
        env_source = _env("lightcurve_source", None, str)
        if env_source is not None:
            resolved_lightcurve_source = env_source
        elif "lightcurve_source" in fc:
            resolved_lightcurve_source = fc["lightcurve_source"]
        else:
            resolved_lightcurve_source = "download"
    resolved_lightcurve_source = _parse_lightcurve_source(resolved_lightcurve_source)

    resolved_lightcurve_fits = lightcurve_fits
    if resolved_lightcurve_fits is None:
        env_fits = _env("lightcurve_fits", None, str)
        if env_fits is not None:
            resolved_lightcurve_fits = env_fits
        elif "lightcurve_fits" in fc:
            resolved_lightcurve_fits = fc["lightcurve_fits"]
        else:
            resolved_lightcurve_fits = None
    resolved_lightcurve_fits = _parse_fits_paths(resolved_lightcurve_fits)

    resolved_target = "" if target is None else str(target).strip()

    resolved_rv_times = _parse_float_array(rv_times)
    resolved_rv_vals = _parse_float_array(rv_vals)
    resolved_rv_errs = _parse_float_array(rv_errs)
    resolved_rv_file = str(rv_file) if rv_file is not None else None

    return PipelineConfig(
        target=resolved_target,
        author=resolve("author", author, str, constants.DEFAULT_AUTHOR),
        cadence=resolve("cadence", cadence, int, constants.DEFAULT_CADENCE),
        sectors=resolved_sectors,
        force_download=force_download,
        lightcurve_source=resolved_lightcurve_source,
        lightcurve_fits=resolved_lightcurve_fits,
        search_method=resolve("search_method", search_method, str, constants.DEFAULT_SEARCH_METHOD),
        period_min=resolve("period_min", period_min, float, constants.DEFAULT_PERIOD_MIN),
        period_max=resolve("period_max", period_max, float, constants.DEFAULT_PERIOD_MAX),
        period_override=period_override,
        max_planets=resolve("max_planets", max_planets, int, constants.DEFAULT_MAX_PLANETS),
        stellar_method=resolve(
            "stellar_method", stellar_method, str, constants.DEFAULT_STELLAR_METHOD
        ),
        inference=resolve("inference", inference, bool, True),
        inference_backend=resolve(
            "inference_backend", inference_backend, str, constants.DEFAULT_INFERENCE_BACKEND
        ),
        chains=resolve("chains", chains, int, constants.DEFAULT_CHAINS),
        draws=resolve("draws", draws, int, constants.DEFAULT_DRAWS),
        tune=resolve("tune", tune, int, constants.DEFAULT_TUNE),
        target_accept=resolve("target_accept", target_accept, float, constants.DEFAULT_TARGET_ACCEPT),
        gp_kernel=resolve("gp_kernel", gp_kernel, str, constants.DEFAULT_GP_KERNEL),
        rv_times=resolved_rv_times,
        rv_vals=resolved_rv_vals,
        rv_errs=resolved_rv_errs,
        rv_file=resolved_rv_file,
        input_is_magnitude=resolve("input_is_magnitude", input_is_magnitude, bool, False),
        output_dir=resolve("output_dir", output_dir, Path, constants.DEFAULT_OUTPUT_DIR),
        plots=resolve("plots", plots, bool, True),
        save_report=save_report,
        verbose=resolve("verbose", verbose, bool, False),
    )
