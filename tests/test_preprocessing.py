def test_preprocess_module_importable() -> None:
    import tess_pipeline.data.preprocess as preprocess

    assert hasattr(preprocess, "preprocess")
