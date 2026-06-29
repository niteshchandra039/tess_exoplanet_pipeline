"""
constants.py — Default pipeline configuration values.

These are the lowest-priority defaults; they are overridden by the user
config file, environment variables, and explicit Pipeline() arguments,
in that order.
"""

from __future__ import annotations

# ─── TESS data acquisition ───────────────────────────────────────────────────

DEFAULT_AUTHOR: str = "SPOC"
DEFAULT_CADENCE: int = 120          # seconds (2-minute short cadence)
DEFAULT_ALL_SECTORS: bool = True
DEFAULT_QUALITY_BITMASK: str = "default"

# ─── Preprocessing ───────────────────────────────────────────────────────────

DEFAULT_SIGMA_CLIP_LOWER: float = 20.0
DEFAULT_SIGMA_CLIP_UPPER: float = 5.0
DEFAULT_FLATTEN_WINDOW_LENGTH: int = 401   # cadences (~13.4 hours for 2-min)
DEFAULT_FLATTEN_POLYORDER: int = 3
DEFAULT_FLATTEN_BREAK_TOLERANCE: int = 5

# ─── Period search ───────────────────────────────────────────────────────────

DEFAULT_SEARCH_METHOD: str = "tls"  # "tls", "bls", or "both"
DEFAULT_PERIOD_MIN: float = 0.5     # days
DEFAULT_PERIOD_MAX: float = 100.0   # days
DEFAULT_DURATION_GRID_STEP: float = 1.05

# ─── Stellar characterization ────────────────────────────────────────────────

DEFAULT_STELLAR_METHOD: str = "isoclassify"   # or "gaia_only"

# ─── Inference ───────────────────────────────────────────────────────────────

DEFAULT_INFERENCE_BACKEND: str = "exoplanet"  # or "batman_only" or "phoebe"
DEFAULT_CHAINS: int = 1
DEFAULT_DRAWS: int = 2
DEFAULT_TUNE: int = 2
DEFAULT_TARGET_ACCEPT: float = 0.9
DEFAULT_GP_KERNEL: str = "SHO"      # or "Matern32"

# ─── Limb darkening ──────────────────────────────────────────────────────────

DEFAULT_LD_LAW: str = "quadratic"   # Kipping (2013) q1/q2 parameterization

# ─── Output / cache ──────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR: str = "output"
DEFAULT_CACHE_SUBDIR: str = ".tess_pipeline_cache"

# ─── Convergence thresholds ──────────────────────────────────────────────────

RHAT_THRESHOLD: float = 1.01
ESS_THRESHOLD: int = 400

# ─── Physical constants ──────────────────────────────────────────────────────

R_SUN_CM: float = 6.957e10          # cm
R_EARTH_CM: float = 6.371e8         # cm
R_EARTH_R_SUN: float = R_EARTH_CM / R_SUN_CM
AU_CM: float = 1.496e13             # cm
