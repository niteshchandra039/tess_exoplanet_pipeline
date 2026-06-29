def test_gaia_module_importable() -> None:
    import tess_pipeline.catalogs.gaia as gaia

    assert hasattr(gaia, "query_gaia")
