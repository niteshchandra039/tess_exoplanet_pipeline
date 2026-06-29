from tess_pipeline.catalogs.stellar import characterize_star


def test_characterize_star_gaia_only_fallback() -> None:
    stellar = characterize_star({"r_star": 1.0, "teff": 5800, "feh": 0.0}, method="gaia_only")
    assert stellar["method"] == "gaia_only"
    assert stellar["r_star"] == 1.0
