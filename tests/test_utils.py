from tess_pipeline.utils.units import r_sun_to_r_earth, stellar_density_cgs


def test_r_sun_to_r_earth_positive() -> None:
    assert r_sun_to_r_earth(1.0) > 100


def test_stellar_density_solar_like() -> None:
    rho = stellar_density_cgs(1.0, 1.0)
    assert 1.0 < rho < 2.0
