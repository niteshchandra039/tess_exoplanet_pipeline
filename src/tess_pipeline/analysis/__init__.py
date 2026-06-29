"""
analysis — Step-by-step TESS exoplanet analysis for notebooks and scripts.

Public API
----------
TESSAnalysis  — main session object; call stages sequentially or ``run()`` for all
"""

from tess_pipeline.analysis.session import TESSAnalysis

__all__ = ["TESSAnalysis"]
