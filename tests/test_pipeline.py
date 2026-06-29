"""Tests for the step-by-step TESSAnalysis session."""

from tess_pipeline import Pipeline, TESSAnalysis
from tess_pipeline.analysis.stages import (
    InferenceStage,
    LightCurveStage,
    PeriodStage,
    StellarStage,
    TargetStage,
    VisualizationStage,
)


def test_tess_analysis_class_exists() -> None:
    assert TESSAnalysis is not None


def test_pipeline_is_alias_for_tess_analysis() -> None:
    assert Pipeline is TESSAnalysis


def test_tess_analysis_exposes_stage_methods() -> None:
    methods = [
        "resolve_target",
        "load_lightcurves",
        "preprocess",
        "lookup_archive_period",
        "search_period",
        "query_gaia",
        "characterize_star",
        "query_sdss",
        "fit_transit",
        "derive_planet_parameters",
        "check_convergence",
        "generate_figures",
        "run",
    ]
    for name in methods:
        assert callable(getattr(TESSAnalysis, name))


def test_stage_modules_exist() -> None:
    for stage in (
        TargetStage,
        LightCurveStage,
        PeriodStage,
        StellarStage,
        InferenceStage,
        VisualizationStage,
    ):
        assert stage is not None
