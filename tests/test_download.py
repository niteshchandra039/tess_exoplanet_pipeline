from __future__ import annotations

from pathlib import Path

import pytest

from tess_pipeline.data.download import load_lightcurves_from_fits
from tess_pipeline.exceptions import DataDownloadError
import tess_pipeline.data.metadata as metadata


def test_resolve_target_accepts_common_formats(monkeypatch) -> None:
    monkeypatch.setattr(metadata, "_fetch_coordinates", lambda tic_id: (None, None))

    a = metadata.resolve_target("TIC 307210830")
    b = metadata.resolve_target("TIC307210830")
    c = metadata.resolve_target(307210830)

    assert a["tic_id"] == 307210830
    assert b["tic_id"] == 307210830
    assert c["tic_id"] == 307210830


def test_resolve_target_invalid_raises(monkeypatch) -> None:
    monkeypatch.setattr(metadata, "_fetch_coordinates", lambda tic_id: (None, None))
    with pytest.raises(Exception):
        metadata.resolve_target("not_a_tic")


def test_load_lightcurves_from_fits_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.fits"
    with pytest.raises(DataDownloadError, match="not found"):
        load_lightcurves_from_fits(missing)


def test_resolve_target_from_fits_reads_header_coordinates() -> None:
    fits_path = Path(__file__).resolve().parents[1] / (
        "data/fits/tess2018206045859-s0001-0000000270501383-0120-s_lc.fits"
    )
    if not fits_path.exists():
        pytest.skip("sample FITS file not available")

    target = metadata.resolve_target_from_fits(fits_path)
    assert target["tic_id"] == 270501383
    assert target["ra"] == pytest.approx(324.535016310363, rel=1e-6)
    assert target["dec"] == pytest.approx(-31.7374835163051, rel=1e-6)
