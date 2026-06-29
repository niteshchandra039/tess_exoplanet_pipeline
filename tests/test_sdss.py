def test_sdss_module_importable() -> None:
    import tess_pipeline.catalogs.sdss as sdss

    assert hasattr(sdss, "query_sdss_rv")
