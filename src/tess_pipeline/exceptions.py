"""
exceptions.py — Custom exception hierarchy for tess_pipeline.
"""

from __future__ import annotations


class TessPipelineError(Exception):
    """Base class for all tess_pipeline errors."""


# ─── Data errors ─────────────────────────────────────────────────────────────


class DataDownloadError(TessPipelineError):
    """Raised when TESS light curve download fails or returns no data."""


class PreprocessingError(TessPipelineError):
    """Raised when light curve preprocessing fails."""


class NoCadenceDataError(DataDownloadError):
    """Raised when no 2-minute cadence SPOC data exists for the target."""


# ─── Target resolution ───────────────────────────────────────────────────────


class TargetResolutionError(TessPipelineError):
    """Raised when a target string cannot be resolved to a TIC ID."""


# ─── Catalog errors ──────────────────────────────────────────────────────────


class CatalogQueryError(TessPipelineError):
    """Raised when an external catalog query fails unexpectedly."""


class GaiaQueryError(CatalogQueryError):
    """Raised when the Gaia DR3 query fails."""


class NASAArchiveError(CatalogQueryError):
    """Raised when the NASA Exoplanet Archive query fails."""


class SDSSQueryError(CatalogQueryError):
    """Raised when the SDSS query fails."""


# ─── Period search ───────────────────────────────────────────────────────────


class PeriodSearchError(TessPipelineError):
    """Raised when period search (TLS / BLS) fails to find a signal."""


# ─── Inference ───────────────────────────────────────────────────────────────


class InferenceError(TessPipelineError):
    """Raised when Bayesian transit fitting fails."""


class InferenceNotInstalledError(InferenceError):
    """Raised when required inference libraries fail to import."""


class ConvergenceWarning(UserWarning):
    """Issued when MCMC chains do not meet R-hat / ESS convergence criteria."""


# ─── Configuration ───────────────────────────────────────────────────────────


class ConfigurationError(TessPipelineError):
    """Raised for invalid or inconsistent configuration values."""
