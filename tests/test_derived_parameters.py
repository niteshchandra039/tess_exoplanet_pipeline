import pytest
import numpy as np
import arviz as az
from tess_pipeline.transit.parameters import derive_planet_parameters

def test_derive_planet_parameters_mock() -> None:
    # 1. Create a mock ArviZ InferenceData posterior trace
    np.random.seed(42)
    draws = 100
    chains = 2
    
    posterior_vars = {
        "rp_r_star": np.random.normal(0.05, 0.001, (chains, draws)),
        "b": np.random.normal(0.2, 0.05, (chains, draws)),
        "period": np.random.normal(5.0, 0.001, (chains, draws)),
        "t0": np.random.normal(120.0, 0.01, (chains, draws)),
        "rho_star": np.random.normal(1.4, 0.1, (chains, draws)),
    }
    
    # ArviZ 1.2.0 expects a dictionary mapping group names to variable dicts
    posterior = az.from_dict({"posterior": posterior_vars})
    
    # 2. Create stellar parameters dict
    stellar = {
        "r_star": 1.0,
        "r_star_err": 0.05,
        "teff": 5777.0,
        "teff_err": 100.0,
    }
    
    # 3. Call parameter derivation
    result = derive_planet_parameters(posterior, stellar)
    
    # 4. Assert key output variables exist
    keys = [
        "rp_r_star", "rp_r_star_err",
        "b", "b_err",
        "period", "period_err",
        "t0", "t0_err",
        "rho_star", "rho_star_err",
        "a_r_star", "a_r_star_err",
        "t14_hr", "t14_hr_err",
        "rp_earth", "rp_earth_err",
        "a_au", "a_au_err",
        "t_eq", "t_eq_err",
    ]
    for key in keys:
        assert key in result, f"Key {key} missing from derived parameters"
        assert result[key] is not None
        assert np.isfinite(result[key])
        
    # Check physical constraints
    assert 0.04 < result["rp_r_star"] < 0.06
    assert 0.1 < result["b"] < 0.3
    assert result["period"] > 0
    assert result["t14_hr"] > 0
    assert result["rp_earth"] > 0
    assert result["a_au"] > 0
    assert result["t_eq"] > 0
