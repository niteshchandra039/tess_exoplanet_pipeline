def test_bayesian_module_importable() -> None:
    import tess_pipeline.inference.bayesian as bayesian

    assert hasattr(bayesian, "run_bayesian_fit")
