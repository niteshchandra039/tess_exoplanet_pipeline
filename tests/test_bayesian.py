from typing import Any


def test_bayesian_module_importable() -> None:
    import tess_pipeline.inference.bayesian as bayesian

    assert hasattr(bayesian, "run_bayesian_fit")


def test_sampling_with_fallback_uses_prior_when_initial_energy_is_bad() -> None:
    import tess_pipeline.inference.bayesian as bayesian

    calls: list[str] = []

    def sample_fn(**kwargs: Any) -> str:
        calls.append(kwargs["init"])
        if kwargs["init"] == "adapt_diag":
            raise RuntimeError("Bad initial energy")
        return "ok"

    result = bayesian._run_sampling_with_fallback(
        sample_fn,
        model=None,
        init_dict={"period": [1.0]},
        draws=10,
        tune=5,
        chains=1,
        target_accept=0.95,
    )

    assert result == "ok"
    assert calls == ["adapt_diag", "prior"]
