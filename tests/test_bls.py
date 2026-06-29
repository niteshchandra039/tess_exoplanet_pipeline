def test_bls_module_importable() -> None:
    import tess_pipeline.transit.bls as bls

    assert hasattr(bls, "run_bls")
