import numpy as np
import pytest
from tess_pipeline.config import build_config
from tess_pipeline.transit.phoebe_model import phoebe_available, run_phoebe_fit


def test_phoebe_available_callable() -> None:
    assert callable(phoebe_available)
    assert callable(run_phoebe_fit)


def test_config_rv_parsing() -> None:
    # Test passing lists of floats for radial velocity
    cfg = build_config(
        "TIC 123456",
        rv_times=[1.0, 2.0, 3.0],
        rv_vals=[10.5, 11.2, 10.9],
        rv_errs=[0.1, 0.2, 0.1],
        input_is_magnitude=True,
    )
    assert cfg.rv_times == (1.0, 2.0, 3.0)
    assert cfg.rv_vals == (10.5, 11.2, 10.9)
    assert cfg.rv_errs == (0.1, 0.2, 0.1)
    assert cfg.input_is_magnitude is True


def test_config_rv_file_parsing(tmp_path) -> None:
    # Test reading from an RV file
    rv_file = tmp_path / "rv_data.csv"
    with open(rv_file, "w") as f:
        f.write("# time, velocity, error\n")
        f.write("100.1, 5.2, 0.05\n")
        f.write("100.2, 5.8, 0.06\n")
        f.write("100.3, 5.5, 0.05\n")

    cfg = build_config(
        "TIC 123456",
        rv_file=rv_file,
    )
    assert len(cfg.rv_times) == 3
    assert np.allclose(cfg.rv_times, [100.1, 100.2, 100.3])
    assert np.allclose(cfg.rv_vals, [5.2, 5.8, 5.5])
    assert np.allclose(cfg.rv_errs, [0.05, 0.06, 0.05])
