import pytest
from tess_pipeline.config import build_config
from tess_pipeline.exceptions import ConfigurationError
from tess_pipeline import TESSAnalysis


def test_parse_sectors_valid() -> None:
    # Test valid sector parsing
    cfg = build_config("TIC 307210830", sectors=1)
    assert cfg.sectors == 1

    cfg = build_config("TIC 307210830", sectors="2")
    assert cfg.sectors == 2

    cfg = build_config("TIC 307210830", sectors="longest")
    assert cfg.sectors == "longest"

    cfg = build_config("TIC 307210830", sectors="[2:3]")
    assert cfg.sectors == "slice:2:3"

    cfg = build_config("TIC 307210830", sectors="2:4")
    assert cfg.sectors == "slice:2:4"

    cfg = build_config("TIC 307210830", sectors="[:3]")
    assert cfg.sectors == "slice::3"

    cfg = build_config("TIC 307210830", sectors=[1, 3])
    assert cfg.sectors == [1, 3]

    cfg = build_config("TIC 307210830", sectors="all")
    assert cfg.sectors == "all"


def test_parse_sectors_invalid() -> None:
    # Test invalid sector inputs
    with pytest.raises(ConfigurationError):
        build_config("TIC 307210830", sectors="invalid")

    with pytest.raises(ConfigurationError):
        build_config("TIC 307210830", sectors="[1:2:3]")

    with pytest.raises(ConfigurationError):
        build_config("TIC 307210830", sectors=-1)


def test_longest_sectors_integration() -> None:
    analysis = TESSAnalysis(
        target="TIC 261136679",
        inference=False,
        sectors="longest",
        verbose=True
    )
    analysis.resolve_target()
    analysis.lookup_archive_period()
    analysis.load_lightcurves()
    
    raw = analysis._lightcurve.raw_collection
    # Verify the longest continuous runs of sectors:
    # Available: [1, 4, 8, 11, 12, 13, 27, 28, 31, 34, 38, 39, 61, 62, 64, 65, 66, 67, 68, 88, 89, 93, 94, 95]
    # Longest consecutive run is 64, 65, 66, 67, 68 (length 5)
    assert len(raw) == 5
    loaded_sectors = sorted([int(lc.meta.get("SECTOR") or lc.sector) for lc in raw])
    assert loaded_sectors == [64, 65, 66, 67, 68]
    assert analysis.results.metadata["sectors_used"] == [64, 65, 66, 67, 68]
