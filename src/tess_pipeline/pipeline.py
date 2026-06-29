"""
pipeline.py — Backward-compatible alias for :class:`~tess_pipeline.analysis.TESSAnalysis`.

Prefer importing ``TESSAnalysis`` directly for step-by-step notebook use.
"""

from __future__ import annotations

from tess_pipeline.analysis.session import TESSAnalysis

Pipeline = TESSAnalysis

__all__ = ["Pipeline", "TESSAnalysis"]
