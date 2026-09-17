from unittest.mock import patch

from tess_pipeline.catalogs.stellar import characterize_star


@patch("tess_pipeline.catalogs.stellar._query_simbad")
@patch("tess_pipeline.catalogs.stellar._query_vizier_tic")
def test_characterize_star_without_gaia_data(mock_query_vizier, mock_query_simbad) -> None:
    mock_query_vizier.return_value = {
        "r_star": 1.1,
        "r_star_err": 0.05,
        "teff": 5900.0,
        "teff_err": 80.0,
        "logg": 4.4,
        "logg_err": 0.05,
        "feh": 0.1,
        "feh_err": 0.04,
        "parallax": 10.0,
    }
    mock_query_simbad.return_value = None

    stellar = characterize_star({"tic_id": 261136679}, method="gaia_only")

    assert stellar["r_star"] == 1.1
    assert stellar["teff"] == 5900.0
    assert stellar["logg"] == 4.4
    assert stellar["feh"] == 0.1
    assert stellar["m_star"] is not None


def test_characterize_star_gaia_only_fallback() -> None:
    stellar = characterize_star({"r_star": 1.0, "teff": 5800, "feh": 0.0}, method="gaia_only")
    assert stellar["method"] == "gaia_only"
    assert stellar["r_star"] == 1.0
