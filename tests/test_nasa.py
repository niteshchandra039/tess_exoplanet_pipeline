def test_nasa_module_importable() -> None:
    import tess_pipeline.catalogs.nasa_archive as na

    assert hasattr(na, "query_archive")


def test_query_archive_uses_local_csv(monkeypatch, tmp_path) -> None:
    from tess_pipeline.catalogs.nasa_archive import query_archive

    csv_path = tmp_path / "TOI_test.csv"
    csv_path.write_text(
        "# comment\n"
        "rowid,toi,tid,tfopwg_disp,pl_pnum,ra,dec,pl_tranmid,pl_orbper,pl_rade,pl_eqt\n"
        "1,123.01,270501383,PC,1,84.123,-12.456,2459000.5,3.14159,2.5,900\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TESS_PIPELINE_ARCHIVE_CSV", str(csv_path))

    result = query_archive(270501383)

    assert result["source"] == "toi-local"
    assert result["period"] == 3.14159
    assert result["epoch"] == 2459000.5
    assert result["rp_earth"] == 2.5
