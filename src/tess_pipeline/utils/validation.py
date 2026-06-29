"""
utils/validation.py — Input validation helpers.
"""

from __future__ import annotations

from typing import Iterable

from tess_pipeline.exceptions import ConfigurationError


def validate_search_method(method: str) -> str:
    """Validate period search method."""
    allowed = {"tls", "bls", "both"}
    if method not in allowed:
        raise ConfigurationError(f"search_method must be one of {sorted(allowed)}; got {method!r}")
    return method


def validate_inference_backend(backend: str) -> str:
    """Validate inference backend name."""
    allowed = {"exoplanet", "batman_only", "phoebe"}
    if backend not in allowed:
        raise ConfigurationError(
            f"inference_backend must be one of {sorted(allowed)}; got {backend!r}"
        )
    return backend


def validate_positive(name: str, value: float | int) -> float | int:
    """Require a strictly positive numeric value."""
    if value <= 0:
        raise ConfigurationError(f"{name} must be > 0; got {value}")
    return value


def validate_nonempty(name: str, values: Iterable) -> None:
    """Require a non-empty iterable."""
    if len(list(values)) == 0:
        raise ConfigurationError(f"{name} must not be empty")
