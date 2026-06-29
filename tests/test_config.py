import pytest

from tess_pipeline.config import build_config
from tess_pipeline.exceptions import ConfigurationError


def test_build_config_defaults() -> None:
    cfg = build_config("TIC 307210830")
    assert cfg.cadence == 120
    assert cfg.search_method in {"tls", "bls", "both"}
    assert cfg.lightcurve_source == "download"
    assert cfg.lightcurve_fits == ()


def test_build_config_fits_source_requires_paths() -> None:
    with pytest.raises(ConfigurationError, match="lightcurve_fits is required"):
        build_config(lightcurve_source="fits")


def test_build_config_fits_source_allows_no_target() -> None:
    cfg = build_config(
        lightcurve_source="fits",
        lightcurve_fits=["a.fits"],
    )
    assert cfg.target == ""
    assert cfg.lightcurve_source == "fits"


def test_build_config_download_requires_target() -> None:
    with pytest.raises(ConfigurationError, match="target is required"):
        build_config()


def test_build_config_parses_fits_paths() -> None:
    cfg = build_config(
        "TIC 307210830",
        lightcurve_source="fits",
        lightcurve_fits=["a.fits", "b.fits"],
    )
    assert cfg.lightcurve_source == "fits"
    assert tuple(str(p) for p in cfg.lightcurve_fits) == ("a.fits", "b.fits")
