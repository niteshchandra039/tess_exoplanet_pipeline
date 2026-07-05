"""Tests for the StellarStage and SIMBAD fallback/characterization logic."""

from unittest.mock import MagicMock, patch
import pytest
from astropy.table import Table

from tess_pipeline.analysis.stages.stellar import StellarStage
from tess_pipeline.catalogs.stellar import characterize_star, _query_simbad


class DummyConfig:
    def __init__(self) -> None:
        self.stellar_method = "gaia_only"


class DummyResults:
    def __init__(self) -> None:
        self.target = {"tic_id": 261136679, "ra": 100.0, "dec": -50.0}
        self.stellar = {}


def test_stellar_stage_initialization() -> None:
    config = DummyConfig()
    results = DummyResults()
    stage = StellarStage(config, results)
    assert stage.config is config
    assert stage.results is results
    assert stage.gaia_params == {}


@patch("tess_pipeline.catalogs.gaia.query_gaia")
def test_stellar_stage_query_gaia(mock_query_gaia: MagicMock) -> None:
    mock_query_gaia.return_value = {
        "r_star": 1.15,
        "teff": 5855.0,
        "parallax": 2.5,
    }
    config = DummyConfig()
    results = DummyResults()
    stage = StellarStage(config, results)
    
    params = stage.query_gaia()
    assert params["r_star"] == 1.15
    assert params["teff"] == 5855.0
    assert params["tic_id"] == 261136679
    assert stage.gaia_params["tic_id"] == 261136679


@patch("tess_pipeline.catalogs.stellar.characterize_star")
@patch("tess_pipeline.catalogs.gaia.query_gaia")
def test_stellar_stage_characterize(mock_query_gaia: MagicMock, mock_characterize_star: MagicMock) -> None:
    mock_query_gaia.return_value = {"r_star": 1.15, "teff": 5855.0}
    mock_characterize_star.return_value = {
        "r_star": 1.15,
        "m_star": 1.05,
        "teff": 5855.0,
        "method": "gaia_only",
    }
    config = DummyConfig()
    results = DummyResults()
    stage = StellarStage(config, results)
    
    stellar = stage.characterize()
    assert stellar["r_star"] == 1.15
    assert stellar["m_star"] == 1.05
    assert results.stellar == stellar


@patch("tess_pipeline.catalogs.stellar._query_simbad")
@patch("tess_pipeline.catalogs.stellar._query_vizier_tic")
def test_characterize_star_simbad_fallback(mock_query_vizier: MagicMock, mock_query_simbad: MagicMock) -> None:
    # Simulate VizieR returning nothing so SIMBAD fills in missing fields
    mock_query_vizier.return_value = None
    
    # Mock SIMBAD returning supplementary values, references, and units
    mock_query_simbad.return_value = {
        "teff": 5800.0,
        "logg": 4.4,
        "feh": -0.1,
        "parallax": 3.0,
        "teff_ref": "2020A&A...642A..31P",
        "logg_ref": "2020A&A...642A..31P",
        "feh_ref": "2020A&A...642A..31P",
        "parallax_ref": "2018A&A...616A...1G",
        "teff_unit": "K",
        "logg_unit": "dex",
        "feh_unit": "dex",
        "parallax_unit": "mas",
    }
    
    gaia_params = {
        "tic_id": 261136679,
        "r_star": 1.0,
        "teff": None,  # Missing in Gaia
        "logg": None,
        "feh": None,
        "parallax": 54.0,  # Prioritized from Gaia
    }
    
    result = characterize_star(gaia_params, method="gaia_only")
    
    # Verify values are merged and derived
    assert result["teff"] == 5800.0
    assert result["logg"] == 4.4
    assert result["feh"] == -0.1
    assert result["r_star"] == 1.0
    assert result["parallax"] == 54.0  # Kept Gaia's parallax
    assert result["reference"] == "Gaia DR3+2020A&A...642A..31P"
    assert result["teff_source"] == "SIMBAD"
    assert result["logg_source"] == "SIMBAD"
    assert result["parallax_source"] == "Gaia"
    assert result["teff_ref"] == "2020A&A...642A..31P"
    assert result["logg_ref"] == "2020A&A...642A..31P"
    assert result["feh_ref"] == "2020A&A...642A..31P"
    assert result["parallax_ref"] == "Gaia DR3"
    assert result["teff_unit"] == "K"
    assert result["logg_unit"] == "dex"
    assert result["feh_unit"] == "dex"
    assert result["parallax_unit"] == "mas"
    assert result["r_star_unit"] == "R_sun"
    assert result["m_star_unit"] == "M_sun"
    assert result["rho_star_unit"] == "g/cm^3"


@patch("tess_pipeline.catalogs.stellar._query_simbad")
@patch("tess_pipeline.catalogs.stellar._query_vizier_tic")
def test_characterize_star_priority(mock_query_vizier: MagicMock, mock_query_simbad: MagicMock) -> None:
    
    # 1. Vizier returns teff, logg
    mock_query_vizier.return_value = {
        "teff": 6000.0,
        "teff_err": 50.0,
        "teff_ref": "TIC8_Teff_Ref",
        "teff_unit": "K",
        "logg": 4.5,
        "logg_err": 0.05,
        "logg_ref": "TIC8_Logg_Ref",
        "logg_unit": "dex",
        "source": "VizieR (TIC v8.2)"
    }
    
    # 2. SIMBAD returns teff, logg, feh
    mock_query_simbad.return_value = {
        "teff": 5800.0,
        "teff_err": 100.0,
        "teff_ref": "SIMBAD_Teff_Ref",
        "teff_unit": "K",
        "logg": 4.4,
        "logg_err": 0.1,
        "logg_ref": "SIMBAD_Logg_Ref",
        "logg_unit": "dex",
        "feh": -0.2,
        "feh_err": 0.05,
        "feh_ref": "2025_Latest_FeH_Ref",
        "feh_unit": "dex",
        "source": "SIMBAD"
    }
    
    # 3. Gaia input
    gaia_params = {
        "tic_id": 261136679,
        "r_star": 1.1,
        "r_star_err": 0.05,
        "teff": 5900.0,    # In Gaia, but Vizier has priority
        "teff_err": 80.0,
        "logg": None,      # Not in Gaia, but Vizier has priority
        "feh": None,       # Not in Gaia or Vizier, SIMBAD has fallback
        "feh_err": None,
        "parallax": 10.0,
        "parallax_err": 0.1
    }
    
    result = characterize_star(gaia_params, method="gaia_only")
    
    # teff should be from Vizier (6000.0)
    assert result["teff"] == 6000.0
    assert result["teff_source"] == "VizieR (TIC8.2)"
    assert result["teff_ref"] == "TIC8_Teff_Ref"
    
    # logg should be from Vizier (4.5)
    assert result["logg"] == 4.5
    assert result["logg_source"] == "VizieR (TIC8.2)"
    assert result["logg_ref"] == "TIC8_Logg_Ref"
    
    # feh should be from SIMBAD (-0.2)
    assert result["feh"] == -0.2
    assert result["feh_source"] == "SIMBAD"
    assert result["feh_ref"] == "2025_Latest_FeH_Ref"


@patch("astroquery.simbad.SimbadClass.query_object")
@patch("astroquery.simbad.SimbadClass.query_tap")
@patch("astroquery.simbad.SimbadClass.add_votable_fields")
def test_query_simbad_by_tic(mock_add_fields: MagicMock, mock_query_tap: MagicMock, mock_query_object: MagicMock) -> None:
    # 1. Mock query_tap returning multiple metallicity measurements (including 2006 and 2025)
    tap_table = Table(
        names=["fe_h", "teff", "log_g", "bibcode"],
        dtype=[float, float, float, str]
    )
    tap_table["fe_h"].unit = "dex"
    tap_table["teff"].unit = "unit-degK"
    tap_table["log_g"].unit = "cm/s**2"
    
    # Old measurement from 2006
    tap_table.add_row([0.11, 6027.0, 4.45, "2006A&A...458..873S"])
    # New measurement from 2025 (latest)
    tap_table.add_row([0.15, 6050.0, 4.48, "2025A&A...789..123X"])
    
    mock_query_tap.return_value = tap_table

    # 2. Mock query_object returning main basic row (e.g. parallax)
    main_table = Table(
        names=["MAIN_ID", "plx_value", "coo_bibcode"],
        dtype=[str, float, str]
    )
    main_table["plx_value"].unit = "mas"
    main_table.add_row(["TIC 261136679", 2.54, "2018A&A...616A...1G"])
    mock_query_object.return_value = main_table

    res = _query_simbad(tic_id=261136679)
    assert res is not None
    # Latest 2025 values should be chosen
    assert res["teff"] == 6050.0
    assert res["logg"] == 4.48
    assert res["feh"] == 0.15
    assert res["parallax"] == 2.54
    assert res["teff_ref"] == "2025A&A...789..123X"
    assert res["logg_ref"] == "2025A&A...789..123X"
    assert res["feh_ref"] == "2025A&A...789..123X"
    assert res["parallax_ref"] == "2018A&A...616A...1G"
    assert res["teff_unit"] == "K"
    assert res["logg_unit"] == "dex"
    assert res["feh_unit"] == "dex"
    assert res["parallax_unit"] == "mas"
    assert res["source"] == "SIMBAD"

