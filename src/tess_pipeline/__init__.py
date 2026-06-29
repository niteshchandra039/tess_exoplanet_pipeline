"""
tess_pipeline — End-to-end TESS exoplanet detection and characterization.

Public API
----------
TESSAnalysis     — step-by-step analysis session (recommended for notebooks)
PipelineResults  — container for all stage outputs
Pipeline         — alias for TESSAnalysis (backward compatibility)
"""

from tess_pipeline.analysis import TESSAnalysis
from tess_pipeline.pipeline import Pipeline
from tess_pipeline.results import PipelineResults

__version__ = "0.1.0"
__all__ = ["TESSAnalysis", "Pipeline", "PipelineResults", "__version__"]
